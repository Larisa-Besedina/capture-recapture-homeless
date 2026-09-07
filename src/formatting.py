"""
Округление таблиц для показа на экране и сохранения.

Промежуточные расчёты ведутся без округления; здесь собраны правила, по
которым числа приводятся к отчётному виду в самый последний момент.

Правила округления (математическое):
  * абсолютная численность людей -> целое;
  * статистики моделей (BIC, AIC, deviance, dev/df) -> 1 знак;
  * коэффициенты модели (coef, SE, p_value) -> 3 знака;
  * всё остальное (доли, проценты) -> 2 знака.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

# Колонки абсолютной численности людей -> целые.
COUNT_COLUMNS = frozenset({
    "obs", "unobs", "est", "ci_lower", "ci_upper",
    "n", "f1", "f2", "f3",
    "chao_est", "nonsat_est", "sat_est",
})

# Колонки, где нужно не 2 знака по умолчанию, а иное число.
DIGITS_OVERRIDE = {
    "bic": 1, "aic": 1, "deviance": 1, "dev_df_ratio": 1,
    "coef": 3, "SE": 3, "p_value": 3,
    "coverage": 1, "coverage_%": 1,   # охват показываем с точностью до 0.1%
}


def round_half_up(x, digits: int = 0):
    """Математическое округление. Сохраняет NaN.

    Нужно вместо встроенного round(), который округляет половины к чётному
    (round(2.5) == 2). Здесь 2.5 -> 3.
    """
    if pd.isna(x):
        return x
    q = Decimal(1).scaleb(-digits)                 # 10^-digits
    return float(Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def _column_digits(col: str) -> int | None:
    """Сколько знаков оставить в колонке при выводе. None -> целое."""
    if col in COUNT_COLUMNS or col.endswith("_n"):   # *_n = число людей в источнике
        return None
    return DIGITS_OVERRIDE.get(col, 2)               # по умолчанию 2 знака


def round_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Копия таблицы, готовая к показу или сохранению.

    Абсолютные численности -> целые (Int64, чтобы NaN не ломал тип),
    остальное -> по правилам выше. Исходная таблица не меняется.
    """
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        nd = _column_digits(col)
        if nd is None:
            out[col] = out[col].map(lambda v: round_half_up(v, 0)).astype("Int64")
        else:
            out[col] = out[col].map(lambda v: round_half_up(v, nd))
    return out
