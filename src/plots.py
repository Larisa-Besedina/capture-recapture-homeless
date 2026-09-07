"""
Графики. Все функции сохраняют файл в pictures/ и возвращают Figure.

Единая палитра PALETTE:

    чёрный   — наблюдаемая численность
    красный  — Чао
    оранжевый— Зельтерман
    синий    — лог-линейная ненасыщенная
    зелёный  — лог-линейная насыщенная

Размеры шрифтов, маркеров и линий собраны в STYLE.

Общие принципы построения:
  * пути к файлам берутся из config;
  * значения выравниваются по списку сезонов через reindex, а не по порядку
    строк: если метода нет в каком-то сезоне, точки иначе съедут на чужой;
  * obs проверяется на согласованность между методами;
  * границы оси считаются через nanmin/nanmax, чтобы один пропуск в ci_lower
    не обнулял весь предел;
  * обрезка оси Y задаётся параметром y_limits, и при срабатывании на панели
    появляется пометка, чтобы усы интервалов не исчезали незаметно.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config

# ----------------------------------------------------------------------
# Единая палитра и стиль
# ----------------------------------------------------------------------
PALETTE = {
    "obs": "#000000",            # наблюдаемая численность
    "obs_fill": "#d9d9d9",       # заливка наблюдаемой части
    "chao": "#d7191c",
    "zelterman": "#fdae61",
    "loglin_nonsaturated": "#2c7bb6",
    "loglin_saturated": "#1b7837",
    "unobs_fill": "#f4a582",     # заливка скрытой части
}

# Размеры подобраны под печатное разрешение (dpi берётся из config).
# Меняйте здесь — значения применяются ко всем рисункам сразу.
STYLE = {
    "font_title": 20,
    "font_label": 16,
    "font_tick": 16,
    "font_tick_x": 15,
    "font_legend": 13,
    "font_legend_big": 22,   # для нижней легенды графика по стратам
    "marker": 12,
    "marker_replace": 18,    # треугольники замены (лог-линейные модели)
    "marker_obs": 10,
    "line_obs": 2.5,
    "capsize": 5,
    "capthick": 2.0,
    "elinewidth": 2.0,
}

# Порядок отрисовки задаёт и порядок записей в легенде
METHOD_ORDER = ["chao", "zelterman", "loglin_nonsaturated", "loglin_saturated"]

METHOD_LABELS = {
    "chao": "Чао",
    "zelterman": "Зельтерман",
    "loglin_nonsaturated": "Лог-линейная ненасыщенная",
    "loglin_saturated": "Лог-линейная насыщенная",
}

METHOD_MARKERS = {
    "chao": "D",
    "zelterman": "^",
    "loglin_nonsaturated": "s",
    "loglin_saturated": "o",
}

# Смещение по оси X, чтобы усы разных методов не накладывались
METHOD_OFFSETS = {
    "loglin_nonsaturated": -0.20,
    "chao": -0.07,
    "zelterman": 0.07,
    "loglin_saturated": 0.20,
}


def save(fig, filename: str, dpi: int | None = None) -> None:
    path = config.PICTURES / filename
    fig.savefig(path, dpi=dpi or config.FIG_DPI, bbox_inches="tight")
    print(f"сохранено: {path.relative_to(config.ROOT)}")


def _observed_by_season(sub: pd.DataFrame, seasons) -> np.ndarray:
    """Наблюдаемая численность по сезонам с проверкой согласованности."""
    agg = sub.groupby("year_season")["obs"].agg(["min", "max"]).reindex(seasons)
    if not np.allclose(agg["min"], agg["max"], equal_nan=True):
        bad = agg[agg["min"] != agg["max"]].index.tolist()
        raise ValueError(
            f"наблюдаемая численность различается между методами в сезонах {bad} — "
            "проверьте, как считается obs"
        )
    return agg["min"].to_numpy(dtype=float)


# ----------------------------------------------------------------------
# Рисунок: все методы для одного уровня агрегации
# ----------------------------------------------------------------------
def plot_method_comparison(all_estimates: pd.DataFrame, strata: str = "total",
                           filename: str = "total_estimates_comparison.png",
                           title: str | None = None):
    """
    Все оценки с доверительными интервалами по сезонам на одной панели.

    Наблюдаемая численность — чёрная линия, оценки — точки с усами со
    смещением по X.
    """
    sub = all_estimates[all_estimates["strata"] == strata].copy()
    if sub.empty:
        raise ValueError(f"в таблице оценок нет строк со strata == {strata!r}")

    seasons = sorted(sub["year_season"].unique())
    x = np.arange(len(seasons), dtype=float)
    methods = [m for m in METHOD_ORDER if m in set(sub["method"])]
    obs = _observed_by_season(sub, seasons)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(x, obs, color=PALETTE["obs"], linewidth=STYLE["line_obs"],
            marker="o", markersize=STYLE["marker_obs"], zorder=10,
            label="Наблюдаемая численность")

    for method in methods:
        # Выравнивание по сезонам, а не по порядку строк
        mdf = sub[sub["method"] == method].set_index("year_season").reindex(seasons)
        y = mdf["est"].to_numpy(dtype=float)
        lo = mdf["ci_lower"].to_numpy(dtype=float)
        hi = mdf["ci_upper"].to_numpy(dtype=float)

        ax.errorbar(
            x + METHOD_OFFSETS.get(method, 0.0), y,
            yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)],
            fmt=METHOD_MARKERS.get(method, "o"),
            color=PALETTE.get(method, "grey"),
            capsize=STYLE["capsize"], capthick=STYLE["capthick"],
            elinewidth=STYLE["elinewidth"],
            markersize=STYLE["marker"], markeredgewidth=1.5,
            markerfacecolor=PALETTE.get(method, "grey"),
            markeredgecolor="white", zorder=5,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=0, ha="center", fontsize=STYLE["font_tick"])
    ax.tick_params(axis="y", labelsize=STYLE["font_tick"])
    ax.set_xlabel("Сезон", fontsize=STYLE["font_label"])
    ax.set_ylabel("Численность", fontsize=STYLE["font_label"])
    if title:
        ax.set_title(title, fontsize=STYLE["font_title"], fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Легенда над осями в три колонки: длинные подписи не помещаются в один
    # ряд по ширине, а над графиком не пересекаются с данными.
    n_labels = len(methods) + 1
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=min(3, n_labels), frameon=True, fontsize=STYLE["font_legend"])

    y_min = np.nanmin([np.nanmin(sub["ci_lower"].to_numpy(dtype=float)), obs.min()])
    y_max = np.nanmax([np.nanmax(sub["ci_upper"].to_numpy(dtype=float)), obs.max()])
    pad = 0.08 * (y_max - y_min)
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)

    fig.tight_layout()
    save(fig, filename)
    return fig


# ----------------------------------------------------------------------
# Рисунок: итоговые оценки по половозрастным группам
# ----------------------------------------------------------------------
def plot_estimates_by_strata(df_final: pd.DataFrame,
                             filename: str = "final_estimates_by_strata.png",
                             y_limits: dict[str, float] | None = None):
    """
    Сетка «пол × возрастная группа»: наблюдаемая численность, итоговая оценка
    и её интервал по сезонам. Маркер и цвет точки показывают, каким методом
    получена итоговая оценка после правила триангуляции.

    y_limits: {"Женщины": 700} — верхняя граница оси Y для всей строки панелей.
    По умолчанию не задана: ось подстраивается под данные. Если задана и
    реальный интервал выходит за границу, на панели появляется пометка
    «ось обрезана», чтобы обрезанные усы не выглядели как узкий интервал.
    """
    y_limits = y_limits or {}
    sex_order = [s for s in config.SEX_ORDER if s in set(df_final["sex"])]
    age_order = [a for a in config.AGE_ORDER if a in set(df_final["age"])]

    # Широкая фигура + мелкий шрифт подписей сезонов (ниже): семь дат вида
    # «2019/2020» на каждой панели размещаются горизонтально без наклона.
    fig, axes = plt.subplots(len(sex_order), len(age_order),
                             figsize=(30, 15), squeeze=False)

    marker_style = {
        "nonsat": (PALETTE["loglin_nonsaturated"], "^"),
        "sat": (PALETTE["loglin_saturated"], "v"),
        None: (PALETTE["chao"], "s"),
    }

    for i, sex in enumerate(sex_order):
        for j, age in enumerate(age_order):
            ax = axes[i][j]
            sub = (df_final[(df_final["sex"] == sex) & (df_final["age"] == age)]
                   .sort_values("year_season"))
            if sub.empty:
                ax.set_axis_off()
                continue

            x = np.arange(len(sub))
            obs = sub["obs"].to_numpy(dtype=float)
            est = sub["est"].to_numpy(dtype=float)
            lo = sub["ci_lower"].to_numpy(dtype=float)
            hi = sub["ci_upper"].to_numpy(dtype=float)
            repl = sub["replaced_by"].to_numpy()

            ax.fill_between(x, 0, obs, color=PALETTE["obs_fill"], alpha=0.5)
            ax.fill_between(x, obs, est, color=PALETTE["unobs_fill"], alpha=0.3)

            ax.plot(x, obs, color=PALETTE["obs"], linewidth=3, marker="o",
                    markersize=STYLE["marker_obs"], markerfacecolor="white",
                    markeredgewidth=2.5, markeredgecolor=PALETTE["obs"], zorder=3)

            for k in range(len(x)):
                key = repl[k] if repl[k] in marker_style else None
                color, marker = marker_style[key]
                # треугольники замены крупнее красного квадрата Чао
                msize = STYLE["marker"] if key is None else STYLE["marker_replace"]
                ax.errorbar(x[k], est[k],
                            yerr=[[max(est[k] - lo[k], 0)], [max(hi[k] - est[k], 0)]],
                            fmt=marker, color=color,
                            capsize=6, capthick=2.5, elinewidth=2.5,
                            markersize=msize, markerfacecolor=color,
                            markeredgewidth=2, markeredgecolor="white", zorder=5)

            ax.set_xticks(x)
            # Подписи сезонов горизонтально; мелкий шрифт, чтобы семь дат
            # уместились без наклона.
            ax.set_xticklabels(sub["year_season"], rotation=0, ha="center",
                               fontsize=STYLE["font_tick_x"])
            ax.set_title(f"{sex}, {age}\n", fontsize=STYLE["font_title"],
                         fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            # Подписи оси Y на всех панелях, название — только в первом столбце
            ax.tick_params(axis="y", labelsize=STYLE["font_tick"])
            if j == 0:
                ax.set_ylabel("Численность\n", fontsize=STYLE["font_label"])
            if i == len(sex_order) - 1:
                ax.set_xlabel("Сезон\n", fontsize=STYLE["font_label"])

    # Единый масштаб по строкам + пометка при обрезке
    for i, sex in enumerate(sex_order):
        row = df_final[df_final["sex"] == sex]
        data_max = float(np.nanmax(row["ci_upper"].to_numpy(dtype=float)))
        cap = y_limits.get(sex)
        y_max = min(data_max, cap) if cap else data_max
        for j in range(len(age_order)):
            axes[i][j].set_ylim(0, y_max * 1.12)
            if cap and data_max > cap:
                axes[i][j].text(0.99, 0.98, "ось обрезана",
                                transform=axes[i][j].transAxes, ha="right",
                                va="top", fontsize=11, style="italic", color="grey")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=PALETTE["obs"], markeredgewidth=2.5,
               markersize=STYLE["marker"], label="Наблюдаемая численность"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["chao"],
               markeredgecolor="white", markeredgewidth=2,
               markersize=STYLE["marker"], label="Оценочная численность по Чао"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=PALETTE["loglin_nonsaturated"],
               markeredgecolor="white", markeredgewidth=2,
               markersize=STYLE["marker_replace"],
               label="Замена на лог-линейную ненасыщенную модель"),
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor=PALETTE["loglin_saturated"],
               markeredgecolor="white", markeredgewidth=2,
               markersize=STYLE["marker_replace"],
               label="Замена на лог-линейную насыщенную модель"),
        plt.Rectangle((0, 0), 1, 1, fc=PALETTE["obs_fill"], alpha=0.5,
                      label="Наблюдаемая численность"),
        plt.Rectangle((0, 0), 1, 1, fc=PALETTE["unobs_fill"], alpha=0.3,
                      label="Скрытая численность"),
    ]

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, hspace=0.35)
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=STYLE["font_legend_big"], frameon=True,
               bbox_to_anchor=(0.5, -0.04))

    # Фигура крупная (30x15"), при 300 dpi png вышел бы избыточно тяжёлым;
    # 200 dpi даёт достаточное для печати качество и вчетверо меньший файл.
    save(fig, filename, dpi=200)
    return fig
