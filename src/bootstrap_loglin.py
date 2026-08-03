"""
Параметрический бутстреп для лог-линейных моделей.

Схема: из подогнанных ожидаемых значений mu генерируются пуассоновские
counts, модель переподгоняется, N пересчитывается. Квантили бутстреп-
распределения дают доверительный интервал.

Агрегаты (пол, total) считаются ВНУТРИ каждой итерации, и уже затем берутся
квантили от суммы: сумма квантилей отдельных страт не равна квантилю их
суммы и завысила бы ширину интервала.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm.auto import tqdm

import config


def _percentiles(values, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    arr = np.asarray(values, dtype=float)
    return (float(np.percentile(arr, 100 * alpha)),
            float(np.percentile(arr, 100 * (1 - alpha))))


def bootstrap_loglinear(model_info: dict, source_vars=None, covariate_vars=None,
                        n_boot: int | None = None, rng=None,
                        level: float | None = None,
                        max_n_factor: float | None = None,
                        min_success_share: float | None = None) -> dict | None:
    """
    Бутстреп-интервалы для одной лог-линейной модели.

    Возвращает {"total": (lo, hi), "strata": {...}, "sex": {...}, "diagnostics": {...}}
    или None, если доля успешных итераций слишком мала.
    """
    source_vars = list(source_vars or config.SOURCE_VARS)
    covariate_vars = list(covariate_vars if covariate_vars is not None else config.COVARIATE_VARS)
    n_boot = config.N_BOOT if n_boot is None else n_boot
    level = config.CI_LEVEL if level is None else level
    max_n_factor = config.BOOT_MAX_N_FACTOR if max_n_factor is None else max_n_factor
    min_success_share = (config.BOOT_MIN_SUCCESS_SHARE if min_success_share is None
                         else min_success_share)
    rng = rng or np.random.default_rng(config.SEED)

    result = model_info["result_object"]
    train_data = model_info["train_data"]
    formula = model_info["formula"]
    observed_total = model_info["obs"]

    # mu привязываем к индексу train_data: выравнивание по позиции строк
    # ненадёжно, если где-то поменяется их порядок.
    mu = result.fittedvalues.reindex(train_data.index)
    if mu.isna().any():
        raise ValueError("fittedvalues не выравниваются с train_data по индексу")
    mu_values = mu.to_numpy()

    # Соответствие страта -> (sex, age) и маски строк
    strata_info = (train_data[["strata"] + covariate_vars]
                   .drop_duplicates().set_index("strata"))
    strata_names = list(strata_info.index)
    strata_masks = {sn: (train_data["strata"] == sn).to_numpy() for sn in strata_names}
    sex_groups = {}
    if "sex" in covariate_vars:
        for sex, sub in strata_info.groupby("sex", observed=True):
            sex_groups[sex] = list(sub.index)

    boot_total: list[float] = []
    boot_strata: dict[str, list[float]] = {sn: [] for sn in strata_names}
    boot_sex: dict[str, list[float]] = {s: [] for s in sex_groups}

    n_failed_fit = 0
    n_rejected = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in tqdm(range(n_boot), desc="  бутстреп", leave=False, smoothing=0):
            y_star = rng.poisson(mu_values)

            boot_data = train_data.copy()
            boot_data["count"] = pd.Series(y_star, index=train_data.index)

            try:
                boot_result = smf.glm(formula=formula, data=boot_data,
                                      family=sm.families.Poisson()).fit()
            except Exception:
                n_failed_fit += 1
                continue

            strata_N = {}
            for sn in strata_names:
                predict_row = {v: 0 for v in source_vars}
                for col in covariate_vars:
                    predict_row[col] = strata_info.loc[sn, col]
                mu_000 = float(
                    boot_result.get_prediction(pd.DataFrame([predict_row])).predicted_mean[0]
                )
                strata_N[sn] = float(boot_data.loc[strata_masks[sn], "count"].sum()) + mu_000

            total_N = sum(strata_N.values())

            if not np.isfinite(total_N) or total_N > max_n_factor * observed_total:
                n_rejected += 1
                continue

            boot_total.append(total_N)
            for sn in strata_names:
                boot_strata[sn].append(strata_N[sn])
            # агрегат по полу считаем на этой же итерации
            for sex, members in sex_groups.items():
                boot_sex[sex].append(sum(strata_N[sn] for sn in members))

    n_success = len(boot_total)
    diagnostics = {
        "n_boot": n_boot,
        "n_success": n_success,
        "n_failed_fit": n_failed_fit,
        "n_rejected_diverged": n_rejected,
        "success_share": n_success / n_boot if n_boot else 0.0,
    }

    # Если успешных итераций слишком мало, интервал ненадёжен и не строится
    if diagnostics["success_share"] < min_success_share:
        print(f"  ! бутстреп ненадёжен: успешных итераций "
              f"{n_success}/{n_boot} (<{min_success_share:.0%}); интервал не рассчитан")
        return {"total": None, "strata": {}, "sex": {}, "diagnostics": diagnostics}

    return {
        "total": _percentiles(boot_total, level),
        "strata": {sn: _percentiles(v, level) for sn, v in boot_strata.items() if v},
        "sex": {s: _percentiles(v, level) for s, v in boot_sex.items() if v},
        "diagnostics": diagnostics,
    }


def collect_loglinear_rows(season, model_info: dict, ci: dict | None,
                           method: str, covariate_vars=None) -> pd.DataFrame:
    """
    Приводит результат одной модели к длинному формату:
    year_season, strata, sex, age, obs, unobs, est, ci_lower, ci_upper, method.
    """
    covariate_vars = list(covariate_vars if covariate_vars is not None else config.COVARIATE_VARS)
    if model_info is None:
        return pd.DataFrame()

    ci = ci or {"total": None, "strata": {}, "sex": {}}
    rows = []

    def _row(strata, sex, age, obs, unobs, est, bounds):
        # отсутствующий интервал — это NaN, а не ноль
        lo, hi = (np.nan, np.nan) if bounds is None else bounds
        return {
            "year_season": season, "strata": strata, "sex": sex, "age": age,
            "obs": obs, "unobs": unobs, "est": est,
            "ci_lower": lo, "ci_upper": hi, "method": method,
        }

    strata_est = model_info["strata_estimates"]

    for sn, vals in strata_est.items():
        if sn == "total":
            continue
        rows.append(_row(sn, vals.get("sex"), vals.get("age"),
                         vals["obs"], vals["unobs"], vals["est"],
                         ci["strata"].get(sn)))

    if "sex" in covariate_vars:
        for sex, bounds in ci["sex"].items():
            members = [v for v in strata_est.values() if v.get("sex") == sex]
            if not members:
                continue
            rows.append(_row(str(sex), sex, "total",
                             sum(v["obs"] for v in members),
                             sum(v["unobs"] for v in members),
                             sum(v["est"] for v in members),
                             bounds))

    rows.append(_row("total", "total", "total",
                     model_info["obs"], model_info["unobs"], model_info["est"],
                     ci["total"]))

    return pd.DataFrame(rows)
