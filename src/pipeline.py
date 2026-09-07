"""
Сборка анализа по всем сезонам.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config
from src import chao_zelterman, loglinear
from src.bootstrap_loglin import bootstrap_loglinear, collect_loglinear_rows


def run_loglinear(df: pd.DataFrame, source_vars=None, covariate_vars=None,
                  n_boot: int | None = None,
                  verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    Полный лог-линейный анализ по всем сезонам.

    Возвращает словарь с тремя таблицами:
      "estimates"    — длинная таблица оценок по стратам/полу/total;
      "models"       — по одной строке на выбранную модель (диагностика);
      "coefficients" — коэффициенты выбранных моделей;
      "top_models"   — топ-N моделей каждого класса по сезонам, для display().
    """
    source_vars = list(source_vars or config.SOURCE_VARS)
    covariate_vars = list(covariate_vars if covariate_vars is not None else config.COVARIATE_VARS)
    n_boot = config.N_BOOT if n_boot is None else n_boot

    estimates, models, coefficients, top_models = [], [], [], []

    for season in tqdm(sorted(df["season"].unique()), desc="сезоны", smoothing=0):
        if verbose:
            print(f"\n=== сезон {season} ===")
        df_season = df[df["season"] == season]

        table, diag = loglinear.model_selection(
            df_season, source_vars, covariate_vars, verbose=verbose
        )
        if table.empty:
            print(f"  ! для сезона {season} не осталось моделей после фильтров")
            continue

        top_sat, top_nonsat = loglinear.select_best(table)

        # Таблицы моделей не печатаем: они шире ячейки ноутбука, и текстовый
        # вывод переносится по строкам. Складываем и возвращаем, чтобы ноутбук
        # показал их через display().
        for subset, kind in ((top_sat, "насыщенные"),
                             (top_nonsat, "ненасыщенные")):
            if subset.empty:
                continue
            block = loglinear.format_model_table(subset)
            block.insert(0, "season", season)
            block.insert(1, "class", kind)
            block.insert(2, "rank", range(1, len(block) + 1))
            top_models.append(block)

        if verbose:
            print(f"  наблюдаемая численность: {int(df_season['count'].sum())}")

        for subset, method in ((top_sat, "loglin_saturated"),
                               (top_nonsat, "loglin_nonsaturated")):
            if subset.empty:
                continue
            best = subset.iloc[0].to_dict()

            coefficients.append(
                loglinear.extract_coefficients(best, season, best["model"])
            )

            # Свой генератор на каждую пару (сезон, метод), чтобы результат
            # не зависел от порядка вызовов.
            rng = np.random.default_rng()
            ci = bootstrap_loglinear(best, source_vars, covariate_vars,
                                     n_boot=n_boot, rng=rng)

            if verbose:
                lo, hi = ci["total"] if ci and ci["total"] else (float("nan"),) * 2
                print(f"  {method:20s} est={best['est']:7.0f} "
                      f"[{lo:.0f}, {hi:.0f}]  BIC={best['bic']:7.1f}")
                print(f"    {best['model']}")

            estimates.append(collect_loglinear_rows(season, best, ci, method,
                                                    covariate_vars))

            row = {k: best[k] for k in ("model", "bic", "aic", "deviance", "df_resid",
                                        "n_params", "n_cells", "dev_df_ratio",
                                        "obs", "unobs", "est", "converged")}
            row.update({
                "season": season,
                "method": method,
                "ci_lower": ci["total"][0] if ci and ci["total"] else np.nan,
                "ci_upper": ci["total"][1] if ci and ci["total"] else np.nan,
                "coverage_%": (best["obs"] / best["est"] * 100) if best["est"] > 0 else np.nan,
                "boot_success_share": ci["diagnostics"]["success_share"] if ci else np.nan,
                "n_models_considered": diag["n_after_filter"],
            })
            models.append(row)

    def _concat(parts):
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    return {
        "estimates": _concat(estimates),
        "models": pd.DataFrame(models),
        "coefficients": _concat(coefficients),
        "top_models": _concat(top_models),
    }


def run_chao_zelterman(df: pd.DataFrame, source_vars=None,
                       n_boot: int | None = None) -> pd.DataFrame:
    """Оценки Чао и Зельтермана по всем сезонам."""
    source_vars = list(source_vars or config.SOURCE_VARS)
    n_boot = config.N_BOOT if n_boot is None else n_boot

    parts = []
    for season in tqdm(sorted(df["season"].unique()), desc="сезоны", smoothing=0):
        rng = np.random.default_rng()
        parts.append(chao_zelterman.bootstrap_chao_zelterman(
            df[df["season"] == season], source_vars, n_boot=n_boot, rng=rng
        ))
    return pd.concat(parts, ignore_index=True)


def combine_estimates(loglin_estimates: pd.DataFrame,
                      cz_estimates: pd.DataFrame) -> pd.DataFrame:
    """Единая длинная таблица всех методов."""
    cols = ["year_season", "strata", "sex", "age", "obs", "unobs", "est",
            "ci_lower", "ci_upper", "method"]
    # Диагностика бутстрепа: доля вырожденных ресемплов у Чао/Зельтермана,
    # доля успешных итераций у лог-линейных моделей
    extra = ["degenerate_share", "n_boot_used"]
    left = loglin_estimates.reindex(columns=cols + extra)
    right = cz_estimates.reindex(columns=cols + extra)
    out = pd.concat([left, right], ignore_index=True)
    return out.sort_values(["year_season", "strata", "method"]).reset_index(drop=True)
