"""
Единая точка конфигурации: пути, имена переменных, константы анализа.

Все пути задаются относительно корня репозитория, поэтому проект работает
одинаково при запуске из notebooks/ и из корня.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Пути
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_RESULT = ROOT / "data" / "result"
PICTURES = ROOT / "pictures"

for _d in (DATA_RAW, DATA_INTERIM, DATA_RESULT, PICTURES):
    _d.mkdir(parents=True, exist_ok=True)

# Входные данные
CONTINGENCY_TABLES = DATA_RAW / "contingency_tables.csv"

# Промежуточные результаты
STRATA_METRICS = DATA_INTERIM / "strata_metrics.csv"
LOGLIN_MODEL_SUMMARY = DATA_INTERIM / "loglinear_model_summary.csv"
LOGLIN_COEFFICIENTS = DATA_INTERIM / "loglinear_coefficients.csv"
LOGLIN_TOP_MODELS = DATA_INTERIM / "loglinear_top_models.csv"
ALL_ESTIMATES = DATA_INTERIM / "all_estimates.csv"

# Итог
FINAL_ESTIMATES = DATA_RESULT / "final_estimates.csv"
FINAL_ESTIMATES_XLSX = DATA_RESULT / "final_estimates.xlsx"

CSV_SEP = ";"

# ----------------------------------------------------------------------
# Переменные
# ----------------------------------------------------------------------
# Источники (списки регистрации). Порядок важен только для читаемости формул.
SOURCE_VARS = ["КС", "НП", "ПО"]

# Ковариаты стратификации
COVARIATE_VARS = ["sex", "age"]

# Базовые (референсные) уровни ковариат. Проверяются на наличие в данных:
# если уровень отсутствует, код падает, а не тихо превращает столбец в NaN.
REFERENCE_LEVELS = {
    "sex": "Мужчины",
    "age": "35‒49",   # ВНИМАНИЕ: тире U+2012 (figure dash), как в исходных данных
}

# Порядок уровней для графиков и таблиц
SEX_ORDER = ["Мужчины", "Женщины"]
AGE_ORDER = ["18‒34", "35‒49", "50‒79"]

# Человекочитаемые обозначения термов в таблицах коэффициентов
TERM_LABELS = {"C(sex)": "П", "C(age)": "В"}

# ----------------------------------------------------------------------
# Параметры анализа
# ----------------------------------------------------------------------
# Генератор случайных чисел сознательно НЕ фиксируется: интервалы в data/
# получены до того, как это решение было принято, и воспроизвести их нельзя.
# Границы интервалов при каждом прогоне будут немного отличаться — масштаб
# расхождения описан в README. Точечные оценки детерминированы.
N_BOOT = 1000             # число бутстреп-итераций
CI_LEVEL = 0.95           # уровень доверительного интервала

# Отбор лог-линейных моделей
MAX_DEV_DF_RATIO = 3.0    # порог сверхдисперсии (deviance / df_resid)
TOP_N_MODELS = 3          # сколько лучших моделей показывать в таблице
REQUIRE_CONVERGED = True  # отбрасывать модели без сходимости IRLS

# Бутстреп лог-линейных моделей: отбрасывать итерации с абсурдными оценками
BOOT_MAX_N_FACTOR = 20.0    # N* > factor * N_obs считается расходимостью
BOOT_MIN_SUCCESS_SHARE = 0.9  # минимальная доля успешных итераций

# Графики
FIG_DPI = 300
