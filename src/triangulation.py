"""
Правило триангуляции: Чао ограничивается сверху и снизу лог-линейными оценками.

Правило:
  * если оценка Чао ниже оценки ненасыщенной лог-линейной модели — берём её;
  * если выше оценки насыщенной модели — берём насыщенную;
  * иначе оставляем Чао.

Это эвристика для сведения трёх оценок к одной, а не статистическая
процедура: номинальное покрытие итогового интервала не гарантировано.
Формулируйте это в README и в тексте статьи прямым текстом.

Реализация:
  * страты и агрегаты (пол, total) обрабатываются одним кодом;
  * предпосылка nonsat <= sat проверяется и помечается флагом
    corridor_invalid, если нарушена («коридор» тогда пуст);
  * пропуски в лог-линейных оценках обрабатываются явно, без опоры на
    поведение сравнений с NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.formatting import round_half_up

METHOD_RU = {
    "chao": "Чао",
    "loglin_nonsaturated": "Лог-линейная ненасыщенная",
    "loglin_saturated": "Лог-линейная насыщенная",
}

KEYS = ["year_season", "strata"]


def _wide(all_estimates: pd.DataFrame) -> pd.DataFrame:
    """Разворачивает длинную таблицу оценок в широкую по методам."""
    keep = KEYS + ["sex", "age", "obs", "est", "ci_lower", "ci_upper"]
    parts = {}
    for method, prefix in (("chao", "chao"),
                           ("loglin_nonsaturated", "nonsat"),
                           ("loglin_saturated", "sat")):
        sub = all_estimates.loc[all_estimates["method"] == method, keep].copy()
        # Иначе merge молча размножит строки
        if sub.duplicated(subset=KEYS).any():
            raise ValueError(
                f"Метод {method} встречается несколько раз для одной пары "
                f"{KEYS} — проверьте входную таблицу оценок"
            )
        sub = sub.rename(columns={"est": f"{prefix}_est",
                                  "ci_lower": f"{prefix}_lo",
                                  "ci_upper": f"{prefix}_hi"})
        parts[prefix] = sub

    base = parts["chao"]
    for prefix in ("nonsat", "sat"):
        cols = KEYS + [f"{prefix}_est", f"{prefix}_lo", f"{prefix}_hi"]
        base = base.merge(parts[prefix][cols], on=KEYS, how="left")
    return base


def apply_triangulation(all_estimates: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Применяет правило замены ко всем строкам (страты, пол, total).

    Возвращает таблицу с итоговыми est / ci_lower / ci_upper / method
    и служебными колонками replaced_by, corridor_invalid.
    """
    wide = _wide(all_estimates)

    def _decide(row):
        chao = (row["chao_est"], row["chao_lo"], row["chao_hi"])
        nonsat = (row["nonsat_est"], row["nonsat_lo"], row["nonsat_hi"])
        sat = (row["sat_est"], row["sat_lo"], row["sat_hi"])

        # Нет лог-линейных оценок — правило неприменимо, остаётся Чао
        if not np.isfinite(nonsat[0]) or not np.isfinite(sat[0]):
            chosen, method, replaced_by = chao, "chao", None
            invalid = False
        else:
            # Предпосылка правила: ненасыщенная модель даёт нижнюю границу
            invalid = nonsat[0] > sat[0]
            if row["chao_est"] < nonsat[0]:
                chosen, method, replaced_by = nonsat, "loglin_nonsaturated", "nonsat"
            elif row["chao_est"] > sat[0]:
                chosen, method, replaced_by = sat, "loglin_saturated", "sat"
            else:
                chosen, method, replaced_by = chao, "chao", None

        return pd.Series({
            "est": chosen[0], "ci_lower": chosen[1], "ci_upper": chosen[2],
            "method": method, "method_ru": METHOD_RU[method],
            "replaced_by": replaced_by, "corridor_invalid": invalid,
        })

    out = pd.concat([wide, wide.apply(_decide, axis=1)], axis=1)

    out["unobs"] = out["est"] - out["obs"]
    out["coverage_%"] = (out["obs"] / out["est"] * 100).where(out["est"] > 0)

    if verbose:
        n_repl = out["replaced_by"].notna().sum()
        print(f"заменено оценок: {n_repl} из {len(out)}")
        for by, cnt in out["replaced_by"].value_counts().items():
            print(f"  на {by}: {cnt}")
        n_invalid = int(out["corridor_invalid"].sum())
        if n_invalid:
            print(f"! в {n_invalid} строках nonsat > sat — 'коридор' пуст, "
                  "правило замены для них плохо определено")

    cols = (KEYS + ["sex", "age", "obs", "unobs", "est", "ci_lower", "ci_upper",
                    "coverage_%", "method", "method_ru", "replaced_by",
                    "corridor_invalid", "chao_est", "nonsat_est", "sat_est"])
    return out[cols].sort_values(KEYS).reset_index(drop=True)


def summary_by_season(all_estimates: pd.DataFrame, strata: str = "total") -> pd.DataFrame:
    """Сводка «оценка (интервал)» по методам для выбранного уровня strata."""
    sub = all_estimates[all_estimates["strata"] == strata]
    rows = []
    for season, group in sub.groupby("year_season", sort=True):
        row = {"season": season}
        for _, r in group.iterrows():
            name = r["method"].replace("loglin_", "")
            # round_half_up вместо f-string :.0f — тот округляет половины к чётному
            est = int(round_half_up(r["est"]))
            lo = int(round_half_up(r["ci_lower"]))
            hi = int(round_half_up(r["ci_upper"]))
            row[name] = f"{est} ({lo}–{hi})"
        rows.append(row)
    out = pd.DataFrame(rows)
    order = ["season", "nonsaturated", "chao", "zelterman", "saturated"]
    return out[[c for c in order if c in out.columns]]
