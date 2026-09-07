"""
Чтение, валидация и сохранение данных.

"""

from __future__ import annotations

import pandas as pd

import config
from src.formatting import round_for_display


REQUIRED_COLUMNS = ["season", "strata", "sex", "age", "count"]


def load_contingency_tables(path=None, source_vars=None) -> pd.DataFrame:
    """Читает таблицы сопряжённости и валидирует их."""
    path = path or config.CONTINGENCY_TABLES
    source_vars = source_vars or config.SOURCE_VARS

    df = pd.read_csv(path, sep=config.CSV_SEP, encoding="utf-8")
    validate_contingency_tables(df, source_vars)
    return df


def validate_contingency_tables(df: pd.DataFrame, source_vars=None) -> None:
    """Проверяет структуру таблиц сопряжённости. Падает при первой проблеме."""
    source_vars = source_vars or config.SOURCE_VARS

    missing = [c for c in REQUIRED_COLUMNS + list(source_vars) if c not in df.columns]
    if missing:
        raise ValueError(f"В таблице нет обязательных колонок: {missing}")

    # 1. Индикаторы источников строго бинарны
    for col in source_vars:
        bad = set(df[col].dropna().unique()) - {0, 1}
        if bad:
            raise ValueError(f"Колонка {col} содержит небинарные значения: {bad}")
        if df[col].isna().any():
            raise ValueError(f"Колонка {col} содержит пропуски")

    # 2. count — неотрицательные целые без пропусков
    if df["count"].isna().any():
        raise ValueError("В колонке count есть пропуски")
    if (df["count"] < 0).any():
        raise ValueError("В колонке count есть отрицательные значения")
    if not (df["count"] % 1 == 0).all():
        raise ValueError("В колонке count есть нецелые значения")

    # 3. Ячейка (0,0,0) ненаблюдаема и должна быть нулевой.
    #    Если там окажется ненулевое число, все оценки n и f1/f2 поедут.
    zero_mask = (df[list(source_vars)] == 0).all(axis=1)
    if (df.loc[zero_mask, "count"] != 0).any():
        raise ValueError(
            "Ячейка (0,0,0) содержит ненулевой count — она ненаблюдаема по определению"
        )

    # 4. Профиль захвата внутри страты не должен повторяться
    key = ["season", "strata"] + list(source_vars)
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        raise ValueError(
            f"Дублирующиеся профили захвата ({int(dup.sum())} строк), "
            f"проверьте {key}"
        )

    # 5. Соответствие strata и (sex, age)
    combos = df.groupby("strata", observed=True)[["sex", "age"]].nunique()
    if (combos > 1).any().any():
        raise ValueError("Одному значению strata соответствует более одной пары (sex, age)")

    # 6. Уровни ковариат совпадают с ожидаемыми в config
    #    (главная защита от подмены тире U+2012 на обычный дефис)
    for var, expected in (("sex", config.SEX_ORDER), ("age", config.AGE_ORDER)):
        actual = set(df[var].dropna().unique())
        unexpected = actual - set(expected)
        if unexpected:
            raise ValueError(
                f"Неожиданные уровни {var}: {unexpected}. Ожидались: {expected}. "
                "Проверьте символ тире и написание."
            )
        ref = config.REFERENCE_LEVELS[var]
        if ref not in actual:
            raise ValueError(f"Референсный уровень {var}={ref!r} отсутствует в данных")


def prepare_for_glm(df: pd.DataFrame, source_vars=None, covariate_vars=None) -> pd.DataFrame:
    """
    Приводит типы к виду, который нужен patsy/statsmodels.

    Референсный уровень ковариаты ставится первым в categories — patsy берёт
    первый уровень за базовый. Возвращается копия: исходный df не меняется.
    """
    source_vars = source_vars or config.SOURCE_VARS
    covariate_vars = covariate_vars if covariate_vars is not None else config.COVARIATE_VARS

    out = df.copy()

    for col in source_vars:
        out[col] = out[col].astype(int)

    for col in covariate_vars:
        ref = config.REFERENCE_LEVELS[col]
        levels = [ref] + sorted(set(out[col].unique()) - {ref})
        out[col] = pd.Categorical(out[col], categories=levels)
        if out[col].isna().any():
            # Ловим ситуацию, когда pd.Categorical тихо превратил значения в NaN
            raise ValueError(
                f"После приведения {col} к Categorical появились NaN — "
                "не совпадают уровни"
            )

    return out


def observed_mask(df: pd.DataFrame, source_vars=None) -> pd.Series:
    """True для наблюдаемых ячеек (то есть всех, кроме (0,0,0))."""
    source_vars = source_vars or config.SOURCE_VARS
    return ~(df[list(source_vars)] == 0).all(axis=1)


def save_table(df: pd.DataFrame, path, round_output: bool = True, **kwargs) -> None:
    """
    Сохраняет таблицу с едиными настройками (разделитель ';', кодировка utf-8)
    и печатает путь. По умолчанию округляет численные колонки для вывода;
    round_output=False сохраняет как есть.
    """
    kwargs.setdefault("sep", config.CSV_SEP)
    kwargs.setdefault("index", False)
    kwargs.setdefault("encoding", "utf-8")
    out = round_for_display(df) if round_output else df
    out.to_csv(path, **kwargs)
    print(f"сохранено: {path.relative_to(config.ROOT)}  ({len(out)} строк)")


def save_table_xlsx(df: pd.DataFrame, path, round_output: bool = True,
                    sheet_name: str = "оценки") -> None:
    """Сохраняет таблицу в xlsx (только данные, без формул).
    """
    out = round_for_display(df) if round_output else df
    out.to_excel(path, index=False, sheet_name=sheet_name)
    print(f"сохранено: {path.relative_to(config.ROOT)}  ({len(out)} строк)")
