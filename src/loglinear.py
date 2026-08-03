"""
Лог-линейные модели для оценки численности скрытой популяции.

Логика: модель Пуассона подгоняется по наблюдаемым ячейкам таблицы
сопряжённости (все, кроме (0,0,0)), затем предсказанное значение для
ячейки (0,0,0) в каждой страте берётся как оценка ненаблюдённых.

"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm.auto import tqdm

import config
from src import formatting
from src.data_io import observed_mask, prepare_for_glm


# ----------------------------------------------------------------------
# 1. Генерация формул
# ----------------------------------------------------------------------
def generate_models(source_vars=None, covariate_vars=None,
                    include_cov_interaction: bool = True) -> dict[str, dict]:
    """
    Все иерархические модели с парными взаимодействиями.

    Возвращает {название: {"formula": str, "source_interactions": frozenset}}.
    Состав взаимодействий между источниками хранится явно, чтобы позже
    отличать насыщенные модели от прочих без разбора текста формулы.
    """
    source_vars = list(source_vars or config.SOURCE_VARS)
    covariate_vars = list(covariate_vars or [])   # None -> пустой список

    pairs = []   # (terms_for_formula, human_name, is_source_pair, pair_key)

    for s1, s2 in itertools.combinations(source_vars, 2):
        pairs.append((f"{s1}:{s2}", f"[{s1}:{s2}]", True, (s1, s2)))

    for s in source_vars:
        for c in covariate_vars:
            label = config.TERM_LABELS.get(f"C({c})", c)
            pairs.append((f"{s}:C({c})", f"[{s}:{label}]", False, None))

    base_terms = source_vars + [f"C({c})" for c in covariate_vars]
    base_formula = "count ~ " + " + ".join(base_terms)

    cov_int_formula = cov_int_name = None
    if include_cov_interaction and len(covariate_vars) >= 2:
        cov_int_formula = " + " + ":".join(f"C({c})" for c in covariate_vars)
        labels = [config.TERM_LABELS.get(f"C({c})", c) for c in covariate_vars]
        cov_int_name = "[" + ":".join(labels) + "]"

    models: dict[str, dict] = {}

    def _add(name, formula, source_pairs):
        models[name] = {"formula": formula, "source_interactions": frozenset(source_pairs)}
        if cov_int_formula:
            models[name + cov_int_name] = {
                "formula": formula + cov_int_formula,
                "source_interactions": frozenset(source_pairs),
            }

    _add("main", base_formula, [])

    for r in range(1, len(pairs) + 1):
        for combo in itertools.combinations(range(len(pairs)), r):
            terms = [pairs[i][0] for i in combo]
            name = "main + " + "".join(pairs[i][1] for i in combo)
            src_pairs = [pairs[i][3] for i in combo if pairs[i][2]]
            _add(name, base_formula + " + " + " + ".join(terms), src_pairs)

    return models


# ----------------------------------------------------------------------
# 2. Подгонка одной модели и оценка N
# ----------------------------------------------------------------------
def fit_model(df: pd.DataFrame, formula: str, source_vars=None, covariate_vars=None,
              source_interactions: frozenset | None = None) -> dict:
    """
    Подгоняет одну лог-линейную модель и возвращает оценку N.

    df должен быть уже подготовлен prepare_for_glm.
    Все числовые характеристики возвращаются БЕЗ округления.
    """
    source_vars = list(source_vars or config.SOURCE_VARS)
    covariate_vars = list(covariate_vars or [])

    obs_mask = observed_mask(df, source_vars)
    train_data = df[obs_mask].copy()
    if train_data.empty:
        raise ValueError("Нет наблюдаемых ячеек после удаления (0,0,0)")

    model = smf.glm(formula=formula, data=train_data, family=sm.families.Poisson())
    result = model.fit()

    obs_total = float(train_data["count"].sum())

    strata_estimates: dict[str, dict] = {}
    unobs_total = 0.0

    if covariate_vars:
        # Уникальные страты вместе с их названием и ковариатами
        strata_cols = ["strata"] + covariate_vars
        unique = df[strata_cols].drop_duplicates()

        for _, srow in unique.iterrows():
            mask = np.ones(len(df), dtype=bool)
            for col in covariate_vars:
                mask &= (df[col] == srow[col]).to_numpy()
            strata_obs = float(df.loc[mask & obs_mask.to_numpy(), "count"].sum())

            predict_row = {v: 0 for v in source_vars}
            for col in covariate_vars:
                predict_row[col] = srow[col]

            mu_000 = float(
                result.get_prediction(pd.DataFrame([predict_row])).predicted_mean[0]
            )

            strata_estimates[srow["strata"]] = {
                "obs": strata_obs,
                "unobs": mu_000,
                "est": strata_obs + mu_000,
                "sex": srow.get("sex"),
                "age": srow.get("age"),
            }
            unobs_total += mu_000
    else:
        predict_row = {v: 0 for v in source_vars}
        unobs_total = float(
            result.get_prediction(pd.DataFrame([predict_row])).predicted_mean[0]
        )
        strata_estimates["total"] = {
            "obs": obs_total, "unobs": unobs_total, "est": obs_total + unobs_total,
        }

    df_resid = int(result.df_resid)

    return {
        "formula": formula,
        "obs": obs_total,
        "unobs": unobs_total,
        "est": obs_total + unobs_total,
        "strata_estimates": strata_estimates,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "deviance": float(result.deviance),
        "df_resid": df_resid,
        "n_params": int(len(result.params)),
        "n_cells": int(len(train_data)),
        "dev_df_ratio": result.deviance / df_resid if df_resid > 0 else np.nan,
        "converged": bool(result.converged),
        # насыщенность определяем по составу парных взаимодействий источников
        "is_sat": (
            len(source_interactions) == len(list(itertools.combinations(source_vars, 2)))
            if source_interactions is not None else np.nan
        ),
        "result_object": result,
        "train_data": train_data,
    }


# ----------------------------------------------------------------------
# 3. Перебор моделей для одного сезона
# ----------------------------------------------------------------------
def model_selection(df_season: pd.DataFrame, source_vars=None, covariate_vars=None,
                    max_dev_df_ratio: float | None = None,
                    require_converged: bool | None = None,
                    verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Перебирает все модели для одного сезона.

    Возвращает (таблица моделей, диагностика перебора).
    Таблица отсортирована по BIC (по неокруглённому значению — FIX).
    """
    source_vars = list(source_vars or config.SOURCE_VARS)
    covariate_vars = list(covariate_vars if covariate_vars is not None else config.COVARIATE_VARS)
    max_dev_df_ratio = config.MAX_DEV_DF_RATIO if max_dev_df_ratio is None else max_dev_df_ratio
    require_converged = config.REQUIRE_CONVERGED if require_converged is None else require_converged

    df = prepare_for_glm(df_season, source_vars, covariate_vars)
    models = generate_models(source_vars, covariate_vars, include_cov_interaction=True)

    fitted, failures = [], {}

    # предупреждения statsmodels подавляем локально, только на время перебора
    # через warnings.filterwarnings("ignore") на весь ноутбук
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, spec in tqdm(models.items(), desc="  перебор моделей",
                               leave=False, smoothing=0):
            try:
                res = fit_model(df, spec["formula"], source_vars, covariate_vars,
                                spec["source_interactions"])
            except Exception as exc:          # тип ошибки сохраняем для сводки
                failures[type(exc).__name__] = failures.get(type(exc).__name__, 0) + 1
                continue
            res["model"] = name
            fitted.append(res)

    diagnostics = {
        "n_cells": int(observed_mask(df, source_vars).sum()),
        "n_formulas": len(models),
        "n_fitted": len(fitted),
        "failures": failures,
    }

    if not fitted:
        diagnostics["n_after_filter"] = 0
        return pd.DataFrame(), diagnostics

    table = pd.DataFrame(fitted)
    n_before = len(table)

    # модели без сходимости IRLS отбрасываем: их оценки ненадёжны
    if require_converged:
        table = table[table["converged"]]
    diagnostics["n_dropped_not_converged"] = n_before - len(table)

    # фильтр по неокруглённому dev_df_ratio, чтобы порог был точным
    table = table[(table["dev_df_ratio"] < max_dev_df_ratio) | table["dev_df_ratio"].isna()]

    table = table.sort_values("bic", kind="mergesort").reset_index(drop=True)
    diagnostics["n_after_filter"] = len(table)

    if verbose:
        print(f"  наблюдаемых ячеек: {diagnostics['n_cells']}")
        print(f"  формул сгенерировано: {diagnostics['n_formulas']}, "
              f"подогнано: {diagnostics['n_fitted']}")
        if failures:
            print(f"  не подогнались: {failures}")
        if diagnostics["n_dropped_not_converged"]:
            print(f"  отброшено без сходимости: {diagnostics['n_dropped_not_converged']}")
        print(f"  после фильтров: {diagnostics['n_after_filter']}")

    return table, diagnostics


def select_best(table: pd.DataFrame, top_n: int | None = None):
    """Топ-N насыщенных и топ-N ненасыщенных моделей по BIC."""
    top_n = config.TOP_N_MODELS if top_n is None else top_n
    if table.empty:
        return table, table
    return (table[table["is_sat"]].head(top_n),
            table[~table["is_sat"].astype(bool)].head(top_n))


# ----------------------------------------------------------------------
# 4. Коэффициенты
# ----------------------------------------------------------------------
def _rename_term(term: str) -> str:
    for src, dst in config.TERM_LABELS.items():
        term = term.replace(src, dst)
    return term.replace("[T.", "[")


def stars(p: float) -> str:
    """Классическая нотация значимости."""
    if not np.isfinite(p):
        return ""
    for threshold, mark in ((0.001, "***"), (0.01, "**"), (0.05, "*"), (0.1, ".")):
        if p < threshold:
            return mark
    return ""


def extract_coefficients(model_info: dict, season, model_label: str | None = None) -> pd.DataFrame:
    """Таблица коэффициентов выбранной модели: терм, оценка, стандартная
    ошибка, p-значение и звёздочки значимости.

    p-значения берём готовыми из statsmodels (result.pvalues): формула
    2*(1 - norm.cdf(|z|)) обнуляется при больших |z| из-за потери точности.
    """
    if model_info is None:
        return pd.DataFrame()

    result = model_info["result_object"]
    label = model_label or model_info.get("model", "")

    rows = [
        {
            "season": season,
            "model": label,
            "parameter": _rename_term(term),
            "coef": result.params[term],
            "SE": result.bse[term],
            "p_value": result.pvalues[term],
            "signif": stars(result.pvalues[term]),
        }
        for term in result.params.index
    ]

    return pd.DataFrame(rows)


# Короткие подписи для показа в ноутбуке. Колонки, одинаковые для всех моделей
# одного сезона (obs, n_cells, converged), в таблицу не попадают: obs печатается
# один раз в сводке по сезону, n_cells — в диагностике перебора, а converged
# всегда True, потому что модели без сходимости отфильтрованы раньше.
_DISPLAY_COLUMNS = {
    "model": "модель",
    "bic": "BIC",
    "aic": "AIC",
    "deviance": "dev",
    "df_resid": "df",
    "n_params": "k",
    "dev_df_ratio": "dev/df",
    "unobs": "скрытых",
    "est": "оценка",
    "coverage": "охват,%",
}


def format_model_table(table: pd.DataFrame, strip_prefix: bool = True) -> pd.DataFrame:
    """
    Компактный вид таблицы моделей.

    Возвращает DataFrame — в ноутбуке его нужно показывать через display(),
    а не print(): текстовый вывод переносится по ширине ячейки и становится
    нечитаемым, HTML-таблица прокручивается по горизонтали.
    """
    out = table.copy()
    out["coverage"] = (out["obs"] / out["est"] * 100).where(out["est"] > 0)

    if strip_prefix:
        # "main + " есть у всех моделей и не несёт информации. Модель без
        # взаимодействий называется просто "main" и не затрагивается.
        out["model"] = out["model"].str.replace(r"^main \+ ", "", regex=True)

    # Округляем по исходным именам колонок (round_for_display смотрит на имена),
    # только потом переименовываем в русские подписи.
    keep = [c for c in _DISPLAY_COLUMNS if c in out.columns]
    out = formatting.round_for_display(out[keep])
    return out.rename(columns=_DISPLAY_COLUMNS)
