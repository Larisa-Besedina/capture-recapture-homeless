"""
Оценки Чао и Зельтермана + непараметрический бутстреп.

Обе оценки опираются только на частоты встречаемости f1 и f2 внутри страты и не
требуют предположения о независимости списков, но при малом f2 неустойчивы.

Используется классическая формула Чао `n + f1²/(2·f2)`:
в исходных данных f2 ≥ 4 во всех стратах, поэтому она определена и
воспроизводима по описательной таблице.

Как считаются оценки и интервалы:
  * точечная оценка — по исходным данным;
  * при f2 = 0 оценка не определена: f2 → 0 означает «скрытых очень много»,
    поэтому такие ресемплы помечаются вырожденными, исключаются из
    перцентилей и подсчитываются (доля в колонке degenerate_share);
  * наблюдённое n берётся напрямую из данных;
  * пол и возраст берутся из колонок таблицы;
  * агрегаты (пол, total) считаются как сумма страт внутри итерации.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config
from src.data_io import observed_mask

# Ниже этого значения f2 классическая формула считается неприменимой
MIN_F2_CLASSIC = 3


# ----------------------------------------------------------------------
# Точечные оценки
# ----------------------------------------------------------------------
def chao_estimate(n: int, f1: int, f2: int) -> float:
    """
    Оценка Чао: n + f1² / (2·f2).

    Возвращает np.nan, если f2 = 0 (формула не определена). NaN здесь —
    сознательный выбор: вырожденный ресемпл не должен молча превращаться
    в конкретное число.
    """
    if n == 0:
        return 0.0
    if f2 == 0:
        return np.nan
    return n + f1 ** 2 / (2 * f2)


def zelterman_estimate(n: int, f1: int, f2: int) -> float:
    """
    Оценка Зельтермана через усечённое распределение Пуассона:
    λ = 2·f2/f1,  N = n / (1 − e^(−λ)).

    При f1 = 0 или f2 = 0 оценка не определена (λ = 0 даёт деление на ноль),
    возвращается np.nan.
    """
    if n == 0:
        return 0.0
    if f1 == 0 or f2 == 0:
        return np.nan
    lam = 2.0 * f2 / f1
    return n / (1.0 - np.exp(-lam))


# ----------------------------------------------------------------------
# Разворачивание таблицы сопряжённости в индивидуальные записи
# ----------------------------------------------------------------------
def expand_to_individuals(df_strata: pd.DataFrame, source_vars=None) -> np.ndarray:
    """
    Массив (n, k) индивидуальных профилей захвата.

    Ячейка (0,0,0) исключается явно: люди с нулём захватов ненаблюдаемы,
    и их попадание в выборку испортило бы и n, и f1/f2.
    """
    source_vars = list(source_vars or config.SOURCE_VARS)
    obs = df_strata[observed_mask(df_strata, source_vars)]

    profiles = obs[source_vars].to_numpy(dtype=int)
    counts = obs["count"].to_numpy(dtype=int)
    if counts.sum() == 0:
        return np.zeros((0, len(source_vars)), dtype=int)
    return np.repeat(profiles, counts, axis=0)


def capture_frequencies(records: np.ndarray) -> tuple[int, int]:
    """(f1, f2) по массиву индивидуальных профилей захвата."""
    if len(records) == 0:
        return 0, 0
    caps = records.sum(axis=1)
    return int(np.sum(caps == 1)), int(np.sum(caps == 2))


def _strata_records(df_season: pd.DataFrame, source_vars) -> tuple[dict, dict]:
    """Индивидуальные записи и метаданные по каждой страте сезона."""
    records, meta = {}, {}
    for sn, group in df_season.groupby("strata", observed=True, sort=True):
        records[sn] = expand_to_individuals(group, source_vars)
        meta[sn] = {"sex": group["sex"].iloc[0], "age": group["age"].iloc[0]}
    return records, meta


# ----------------------------------------------------------------------
# Непараметрический бутстреп
# ----------------------------------------------------------------------
def bootstrap_chao_zelterman(df_season: pd.DataFrame, source_vars=None,
                             n_boot: int | None = None, rng=None,
                             level: float | None = None) -> pd.DataFrame:
    """
    Стратифицированный непараметрический бутстреп для Чао и Зельтермана.

    Внутри каждой итерации страты ресемплируются независимо, агрегаты
    считаются как сумма страт на той же итерации — только так интервал
    агрегата корректен (сумма квантилей ≠ квантиль суммы).

    Итерации, в которых оценка не определена (f2* = 0), исключаются из
    расчёта перцентилей; их доля возвращается в колонке degenerate_share.
    Для агрегата итерация исключается, если вырождена хотя бы одна входящая
    в него страта.

    Возвращает длинную таблицу: year_season, strata, sex, age, obs, unobs,
    est, ci_lower, ci_upper, method, degenerate_share, n_boot_used.
    """
    source_vars = list(source_vars or config.SOURCE_VARS)
    n_boot = config.N_BOOT if n_boot is None else n_boot
    level = config.CI_LEVEL if level is None else level
    if rng is None:
        rng = np.random.default_rng()   # не фиксируется, см. config.py
    alpha = (1 - level) / 2

    season = df_season["season"].iloc[0]
    records, meta = _strata_records(df_season, source_vars)
    strata_names = list(records)
    n_strata = len(strata_names)

    # --- точечные оценки по исходным данным ---
    point = {}
    for sn in strata_names:
        n = len(records[sn])
        f1, f2 = capture_frequencies(records[sn])
        if f2 < MIN_F2_CLASSIC:
            warnings.warn(
                f"{season} / {sn}: f2 = {f2} < {MIN_F2_CLASSIC}. Формула Чао "
                "здесь неустойчива, оценку для этой страты трактуйте осторожно.",
                stacklevel=2,
            )
        point[sn] = {
            "n": n,
            "chao": chao_estimate(n, f1, f2),
            "zelterman": zelterman_estimate(n, f1, f2),
        }

    # --- бутстреп: матрицы (n_boot, n_strata), NaN = вырожденная итерация ---
    boot = {m: np.full((n_boot, n_strata), np.nan) for m in ("chao", "zelterman")}

    for it in tqdm(range(n_boot), desc="  бутстреп Чао/Зельтерман",
                   leave=False, smoothing=0):
        for j, sn in enumerate(strata_names):
            rec = records[sn]
            n = len(rec)
            if n == 0:
                boot["chao"][it, j] = 0.0
                boot["zelterman"][it, j] = 0.0
                continue
            resample = rec[rng.integers(0, n, size=n)]
            f1, f2 = capture_frequencies(resample)
            boot["chao"][it, j] = chao_estimate(n, f1, f2)
            boot["zelterman"][it, j] = zelterman_estimate(n, f1, f2)

    # --- сборка результата ---
    def _summarise(values: np.ndarray, n_obs: float, est: float):
        """Перцентили по невырожденным итерациям + доля вырожденных."""
        finite = values[np.isfinite(values)]
        share = 1.0 - len(finite) / len(values) if len(values) else np.nan
        if len(finite) == 0:
            return np.nan, np.nan, share, 0
        lo = float(np.percentile(finite, 100 * alpha))
        hi = float(np.percentile(finite, 100 * (1 - alpha)))
        return max(lo, n_obs), hi, share, len(finite)   # N не меньше наблюдённого

    def _row(strata, sex, age, n_obs, est, values):
        lo, hi, share, used = _summarise(values, n_obs, est)
        return {
            "year_season": season, "strata": strata, "sex": sex, "age": age,
            "obs": n_obs,
            "unobs": max(est - n_obs, 0.0) if np.isfinite(est) else np.nan,
            "est": est, "ci_lower": lo, "ci_upper": hi,
            "degenerate_share": share, "n_boot_used": used,
        }

    sex_levels = sorted({meta[sn]["sex"] for sn in strata_names})
    rows = []

    for m in ("chao", "zelterman"):
        mat = boot[m]

        for j, sn in enumerate(strata_names):
            r = _row(sn, meta[sn]["sex"], meta[sn]["age"],
                     point[sn]["n"], point[sn][m], mat[:, j])
            r["method"] = m
            rows.append(r)

        for s in sex_levels:
            idx = [j for j, sn in enumerate(strata_names) if meta[sn]["sex"] == s]
            # Итерация годится, только если ни одна входящая страта не вырождена
            sums = np.where(np.isfinite(mat[:, idx]).all(axis=1),
                            mat[:, idx].sum(axis=1), np.nan)
            r = _row(str(s), s, "total",
                     sum(point[strata_names[j]]["n"] for j in idx),
                     sum(point[strata_names[j]][m] for j in idx),
                     sums)
            r["method"] = m
            rows.append(r)

        sums = np.where(np.isfinite(mat).all(axis=1), mat.sum(axis=1), np.nan)
        r = _row("total", "total", "total",
                 sum(point[sn]["n"] for sn in strata_names),
                 sum(point[sn][m] for sn in strata_names),
                 sums)
        r["method"] = m
        rows.append(r)

    out = pd.DataFrame(rows)

    worst = out["degenerate_share"].max()
    if worst > 0.01:
        bad = out.loc[out["degenerate_share"] > 0.01, ["strata", "method", "degenerate_share"]]
        print(f"  ! {season}: вырожденные ресемплы (f2*=0) более 1% итераций:")
        for _, b in bad.iterrows():
            print(f"      {b['strata']:20s} {b['method']:10s} {b['degenerate_share']:.1%}")

    return out
