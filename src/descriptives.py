"""
Предварительный анализ: описательные метрики по стратам.

Считаются частоты встречаемости f1/f2/f3, охват каждого источника и попарные
пересечения. Все величины возвращаются без округления — округление
происходит только на этапе отображения (format_metrics, formatting.py).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

import config
from src import formatting


def strata_metrics(df_strata: pd.DataFrame, source_vars=None) -> dict:
    """
    Метрики одной страты по таблице сопряжённости.
    """
    source_vars = list(source_vars or config.SOURCE_VARS)

    obs = df_strata[~(df_strata[source_vars] == 0).all(axis=1)]
    n_total = int(obs["count"].sum())

    cap_sum = obs[source_vars].sum(axis=1)
    freqs = {k: int(obs.loc[cap_sum == k, "count"].sum()) for k in (1, 2, 3)}
    f1, f2, f3 = freqs[1], freqs[2], freqs[3]

    metrics = {
        "n": n_total,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f1_share": f1 / n_total if n_total > 0 else np.nan,
        "f1_f2": f1 / f2 if f2 > 0 else np.nan,
    }

    for source in source_vars:
        n_source = int(obs.loc[obs[source] == 1, "count"].sum())
        metrics[f"{source}_n"] = n_source
        metrics[f"{source}_share"] = n_source / n_total if n_total > 0 else np.nan

    for s1, s2 in combinations(source_vars, 2):
        n12 = int(obs.loc[(obs[s1] == 1) & (obs[s2] == 1), "count"].sum())
        min_n = min(metrics[f"{s1}_n"], metrics[f"{s2}_n"])
        metrics[f"{s1}_{s2}"] = n12
        # Доля меньшего источника, покрытая большим
        metrics[f"{s1}_{s2}_ovl"] = n12 / min_n if min_n > 0 else np.nan

    return metrics


def all_strata_metrics(df: pd.DataFrame, source_vars=None) -> pd.DataFrame:
    """Метрики по всем сезонам и стратам."""
    source_vars = list(source_vars or config.SOURCE_VARS)

    rows = []
    for (season, strata), group in df.groupby(["season", "strata"], observed=True, sort=True):
        m = strata_metrics(group, source_vars)
        m["season"] = season
        m["strata"] = strata
        rows.append(m)

    out = pd.DataFrame(rows)
    front = ["season", "strata"]
    return out[front + [c for c in out.columns if c not in front]]


def format_metrics(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Готовит таблицу метрик к показу: целые для счётчиков, 2 знака для долей."""
    return formatting.round_for_display(df_metrics)


def metrics_summary(df_metrics: pd.DataFrame, source_vars=None) -> pd.DataFrame:
    """Сводка по ключевым метрикам (аналог describe без count и std)."""
    source_vars = list(source_vars or config.SOURCE_VARS)

    cols = ["n", "f1_share", "f1_f2"]
    cols += [f"{s}_share" for s in source_vars]
    cols += [f"{s1}_{s2}" for s1, s2 in combinations(source_vars, 2)]
    cols = [c for c in cols if c in df_metrics.columns]

    desc = df_metrics[cols].describe().drop(index=["count", "std"])
    return formatting.round_for_display(desc)
