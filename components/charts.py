"""Constructores reutilizables de visualizaciones del dashboard.

El modulo concentra las decisiones visuales del proyecto: paleta accesible,
muestreo reproducible, unidades, textos emergentes y configuracion de Plotly.
Las paginas reciben figuras listas para mostrar y no necesitan conocer detalles
de estilo ni repetir transformaciones graficas.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import dendrogram

from config.settings import RANDOM_STATE, SCATTER_SAMPLE_SIZE
from config.theme import (
    CATEGORICAL_ALL_PAIRS,
    CATEGORICAL_FILL,
    CATEGORICAL_INK,
    COLORBAR_STYLE,
    DIVERGING_THERMAL,
    FROST_INTENSITY_COLORS,
    PALETTE,
    PLOTLY_CONFIG,
    PLOTLY_TEMPLATE,
    RISK_LEVEL_COLORS,
    SEQUENTIAL_BLUE,
    SEQUENTIAL_ORANGE,
    hex_to_rgba,
    register_theme,
)
from utils.formatting import variable_label
from utils.stats_tools import fit_linear_trend, mann_kendall_test


_GENERATOR = np.random.default_rng(RANDOM_STATE)
_MONTH_COLUMNS = ("mes_abrev", "mes_nombre", "mes")
_FROST_ORDER = tuple(FROST_INTENSITY_COLORS)

register_theme()


def _label(name: str | None, fallback: str = "Valor") -> str:
    """Devuelve la etiqueta humana de una variable."""
    if not name:
        return fallback
    return variable_label(str(name))


def _colorbar(title: str) -> dict[str, Any]:
    """Copia la configuracion comun sin mutar la constante del tema."""
    result = deepcopy(COLORBAR_STYLE)
    result["title"]["text"] = title
    return result


def _finish(
    figure: go.Figure,
    title: str,
    *,
    height: int | None = None,
    showlegend: bool | None = None,
) -> go.Figure:
    """Aplica la plantilla y ajustes comunes a una figura."""
    layout: dict[str, Any] = {"template": PLOTLY_TEMPLATE, "title": {"text": title}}
    if height is not None:
        layout["height"] = height
    if showlegend is not None:
        layout["showlegend"] = showlegend
    figure.update_layout(**layout)
    return figure


def _empty_figure(title: str, message: str = "No hay datos para la seleccion actual.") -> go.Figure:
    """Crea un estado vacio que sigue siendo una figura Plotly valida."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": PALETTE["ink_muted"]},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _finish(figure, title, height=330, showlegend=False)


def _sample_positions(length: int, sample: int | None) -> np.ndarray:
    """Elige posiciones reproducibles para una visualizacion densa."""
    limit = sample or SCATTER_SAMPLE_SIZE
    if length <= limit:
        return np.arange(length)
    generator = np.random.default_rng(RANDOM_STATE)
    return np.sort(generator.choice(length, size=limit, replace=False))


def _sample_frame(frame: pd.DataFrame, sample: int | None) -> tuple[pd.DataFrame, bool]:
    positions = _sample_positions(len(frame), sample)
    return frame.iloc[positions].copy(), len(positions) < len(frame)


def _series_values(values: Any, row_index: pd.Index, length: int) -> np.ndarray | None:
    """Alinea una serie externa con las observaciones de un resultado."""
    if values is None:
        return None
    if isinstance(values, pd.Series):
        try:
            aligned = values.reindex(row_index)
            if len(aligned) == length:
                return aligned.to_numpy()
        except (KeyError, ValueError):
            aligned = None
        values = values.to_numpy()
    array = np.asarray(values)
    return array if len(array) == length else None


def _collapse_categories(values: pd.Series, maximum: int) -> pd.Series:
    """Conserva las categorias frecuentes y agrupa el resto como ``Otras``."""
    text = values.astype("string").fillna("Sin dato")
    counts = text.value_counts(dropna=False)
    if len(counts) <= maximum:
        return text
    keep = set(counts.head(maximum - 1).index.astype(str))
    return text.map(lambda value: value if str(value) in keep else "Otras")


def _category_order(series: pd.Series, order: Sequence[Any] | None = None) -> list[str]:
    if order is not None:
        present = set(series.astype(str))
        return [str(item) for item in order if str(item) in present]
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna().astype(str))
        return [str(item) for item in series.cat.categories if str(item) in present]
    return [str(item) for item in pd.unique(series.dropna().astype(str))]


def _subtitle_for_sample(title: str, sampled: bool, shown: int, total: int) -> str:
    if not sampled:
        return title
    return f"{title}<br><sup>Muestra reproducible de {shown:,} de {total:,} observaciones (semilla {RANDOM_STATE})</sup>"


def render(figure: go.Figure, *, height: int | None = None, key: str | None = None) -> None:
    """Muestra una figura con la configuracion comun del dashboard."""
    if height is not None:
        figure.update_layout(height=height)
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key=key)


def bar_ranking(
    frame: pd.DataFrame,
    *,
    category: str,
    value: str,
    title: str,
    orientation: str = "h",
    value_label: str | None = None,
    top_n: int | None = None,
    show_values: bool = True,
    color_by_value: bool = False,
) -> go.Figure:
    """Grafica un ranking de categorias ordenado por una magnitud."""
    if frame.empty or category not in frame or value not in frame:
        return _empty_figure(title)
    data = frame.loc[:, [category, value]].dropna().copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna(subset=[value]).sort_values(value, ascending=False)
    if top_n:
        data = data.head(top_n)
    if orientation == "h":
        data = data.sort_values(value, ascending=True)
    if data.empty:
        return _empty_figure(title)

    horizontal = orientation == "h"
    marker: dict[str, Any]
    if color_by_value:
        marker = {
            "color": data[value],
            "colorscale": list(SEQUENTIAL_ORANGE),
            "colorbar": _colorbar(value_label or _label(value)),
        }
    else:
        marker = {"color": CATEGORICAL_FILL[0]}
    text_values = data[value].map(lambda number: f"{number:,.1f}") if show_values else None
    figure = go.Figure(
        go.Bar(
            x=data[value] if horizontal else data[category].astype(str),
            y=data[category].astype(str) if horizontal else data[value],
            orientation="h" if horizontal else "v",
            marker=marker,
            text=text_values,
            textposition="outside" if show_values else "none",
            customdata=data[[category, value]].to_numpy(),
            hovertemplate=(
                f"<b>%{{customdata[0]}}</b><br>{value_label or _label(value)}: "
                "%{customdata[1]:,.2f}<extra></extra>"
            ),
            name=value_label or _label(value),
        )
    )
    figure.update_xaxes(title_text=value_label or (_label(value) if horizontal else _label(category)))
    figure.update_yaxes(title_text=_label(category) if horizontal else (value_label or _label(value)))
    return _finish(figure, title, height=max(380, 34 * len(data) + 130) if horizontal else 430, showlegend=False)


def bar_grouped(
    frame: pd.DataFrame,
    *,
    category: str,
    value: str,
    series: str,
    title: str,
    value_label: str | None = None,
) -> go.Figure:
    """Compara una magnitud por categoria y serie mediante barras agrupadas."""
    required = {category, value, series}
    if frame.empty or not required.issubset(frame.columns):
        return _empty_figure(title)
    data = frame.loc[:, [category, value, series]].dropna().copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna(subset=[value])
    data[series] = _collapse_categories(data[series], len(CATEGORICAL_FILL))
    groups = _category_order(data[series])
    figure = go.Figure()
    for index, group in enumerate(groups):
        subset = data[data[series].astype(str) == group]
        figure.add_bar(
            x=subset[category].astype(str),
            y=subset[value],
            name=group,
            marker_color=CATEGORICAL_FILL[index],
            customdata=subset[[category, value]].to_numpy(),
            hovertemplate=(
                f"<b>{_label(series, series)}: {group}</b><br>"
                f"{_label(category, category)}: %{{customdata[0]}}<br>"
                f"{value_label or _label(value)}: %{{customdata[1]:,.2f}}<extra></extra>"
            ),
        )
    figure.update_layout(barmode="group")
    figure.update_xaxes(title_text=_label(category))
    figure.update_yaxes(title_text=value_label or _label(value))
    return _finish(figure, title, height=440, showlegend=len(groups) > 1)


def bar_stacked_intensity(
    frame: pd.DataFrame,
    *,
    category: str = "provincia",
    title: str = "",
    normalize: bool = False,
) -> go.Figure:
    """Muestra la composicion ordinal de intensidad de helada."""
    intensity = "intensidad_helada"
    if frame.empty or category not in frame or intensity not in frame:
        return _empty_figure(title)
    counts = (
        frame.loc[:, [category, intensity]]
        .dropna()
        .assign(**{category: lambda item: item[category].astype(str), intensity: lambda item: item[intensity].astype(str)})
        .groupby([category, intensity], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    order = [name for name in _FROST_ORDER if name in counts.columns]
    counts = counts.reindex(columns=order)
    if normalize:
        denominator = counts.sum(axis=1).replace(0, np.nan)
        values = counts.div(denominator, axis=0).mul(100).fillna(0)
        suffix = "%"
        axis_title = "Composicion de dias (%)"
    else:
        values = counts
        suffix = " dias"
        axis_title = "Dias analizados"

    figure = go.Figure()
    for name in order:
        figure.add_bar(
            x=values.index,
            y=values[name],
            name=name,
            marker_color=FROST_INTENSITY_COLORS[name],
            customdata=np.column_stack((counts[name].to_numpy(), values[name].to_numpy())),
            hovertemplate=(
                f"<b>%{{x}}</b><br>Intensidad: {name}<br>"
                f"Registros: %{{customdata[0]:,.0f}}<br>Valor: %{{customdata[1]:,.1f}}{suffix}<extra></extra>"
            ),
        )
    figure.update_layout(barmode="stack")
    figure.update_xaxes(title_text=_label(category))
    figure.update_yaxes(title_text=axis_title, range=[0, 100] if normalize else None)
    return _finish(figure, title, height=450, showlegend=len(order) > 1)


def line_series(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    y_label: str | None = None,
    series: str | None = None,
    annotate_last: bool = True,
    fill: bool = False,
) -> go.Figure:
    """Grafica una o varias series temporales o secuenciales."""
    required = {x, y} | ({series} if series else set())
    if frame.empty or not required.issubset(frame.columns):
        return _empty_figure(title)
    data = frame.loc[:, list(required)].dropna(subset=[x, y]).copy()
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[y]).sort_values(x)
    if data.empty:
        return _empty_figure(title)

    if series:
        data[series] = _collapse_categories(data[series], len(CATEGORICAL_INK))
        if data.duplicated([x, series]).any():
            data = data.groupby([x, series], observed=True, as_index=False)[y].mean()
        groups = _category_order(data[series])
    else:
        groups = [""]

    figure = go.Figure()
    for index, group in enumerate(groups):
        subset = data if series is None else data[data[series].astype(str) == group]
        figure.add_scatter(
            x=subset[x],
            y=subset[y],
            mode="lines",
            name=group or y_label or _label(y),
            line={"width": 2, "color": CATEGORICAL_INK[index]},
            fill="tozeroy" if fill else None,
            fillcolor=hex_to_rgba(CATEGORICAL_INK[index], 0.12) if fill else None,
            hovertemplate=(
                f"{_label(x)}: %{{x}}<br>{y_label or _label(y)}: %{{y:,.2f}}"
                + (f"<br>{_label(series)}: {group}" if series else "")
                + "<extra></extra>"
            ),
        )
        if annotate_last and not subset.empty:
            last = subset.iloc[-1]
            figure.add_annotation(
                x=last[x],
                y=last[y],
                text=f"{float(last[y]):,.1f}",
                showarrow=True,
                arrowhead=0,
                ax=22,
                ay=0,
                font={"size": 11, "color": CATEGORICAL_INK[index]},
                arrowcolor=CATEGORICAL_INK[index],
            )
    figure.update_xaxes(title_text=_label(x))
    figure.update_yaxes(title_text=y_label or _label(y))
    return _finish(figure, title, height=430, showlegend=len(groups) > 1)


def line_monthly_cycle(
    monthly: pd.DataFrame,
    *,
    value: str,
    title: str,
    y_label: str | None = None,
    secondary_value: str | None = None,
) -> go.Figure:
    """Grafica el ciclo anual; una segunda magnitud ocupa un panel inferior."""
    if monthly.empty or value not in monthly:
        return _empty_figure(title)
    month_col = next((column for column in _MONTH_COLUMNS if column in monthly), None)
    if month_col is None:
        return _empty_figure(title, "El resumen mensual no contiene una columna de mes.")
    data = monthly.copy()
    if "mes" in data:
        data = data.sort_values("mes")
    x_values = data[month_col]
    has_secondary = bool(secondary_value and secondary_value in data)
    rows = 2 if has_secondary else 1
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=has_secondary,
        vertical_spacing=0.12,
        row_heights=[0.62, 0.38] if has_secondary else None,
    )
    figure.add_scatter(
        x=x_values,
        y=data[value],
        mode="lines+markers",
        name=y_label or _label(value),
        line={"width": 2, "color": CATEGORICAL_INK[0]},
        marker={"size": 9, "color": CATEGORICAL_INK[0], "line": {"width": 1.5, "color": PALETTE["surface"]}},
        hovertemplate=f"Mes: %{{x}}<br>{y_label or _label(value)}: %{{y:,.2f}}<extra></extra>",
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text=y_label or _label(value), row=1, col=1)
    if has_secondary and secondary_value is not None:
        figure.add_scatter(
            x=x_values,
            y=data[secondary_value],
            mode="lines+markers",
            name=_label(secondary_value),
            line={"width": 2, "color": CATEGORICAL_INK[1]},
            marker={"size": 9, "color": CATEGORICAL_INK[1], "line": {"width": 1.5, "color": PALETTE["surface"]}},
            hovertemplate=f"Mes: %{{x}}<br>{_label(secondary_value)}: %{{y:,.2f}}<extra></extra>",
            row=2,
            col=1,
        )
        figure.update_yaxes(title_text=_label(secondary_value), row=2, col=1)
    figure.update_xaxes(title_text="Mes", row=rows, col=1)
    return _finish(figure, title, height=580 if has_secondary else 420, showlegend=has_secondary)


def line_annual_with_trend(
    annual: pd.DataFrame,
    *,
    value: str,
    title: str,
    y_label: str | None = None,
) -> go.Figure:
    """Superpone tendencia OLS y pendiente robusta de Sen a la serie anual."""
    if annual.empty or value not in annual:
        return _empty_figure(title)
    x_column = "anio" if "anio" in annual else annual.columns[0]
    data = annual.loc[:, [x_column, value]].dropna().sort_values(x_column)
    if data.empty:
        return _empty_figure(title)
    x_numeric = pd.to_numeric(data[x_column], errors="coerce")
    y_numeric = pd.to_numeric(data[value], errors="coerce")
    valid = x_numeric.notna() & y_numeric.notna()
    x_numeric, y_numeric = x_numeric[valid], y_numeric[valid]
    figure = go.Figure()
    figure.add_scatter(
        x=x_numeric,
        y=y_numeric,
        mode="lines+markers",
        name="Serie observada",
        line={"width": 2, "color": CATEGORICAL_INK[0]},
        marker={"size": 9, "line": {"width": 1.5, "color": PALETTE["surface"]}},
        hovertemplate=f"Ano: %{{x:.0f}}<br>{y_label or _label(value)}: %{{y:,.2f}}<extra></extra>",
    )
    trend = fit_linear_trend(x_numeric.to_numpy(), y_numeric.to_numpy())
    if trend is not None:
        predicted = trend.intercept + trend.slope * x_numeric.to_numpy()
        figure.add_scatter(
            x=x_numeric,
            y=predicted,
            mode="lines",
            name=f"OLS ({trend.slope:+.2f}/ano; R²={trend.r_squared:.2f})",
            line={"width": 2, "dash": "dash", "color": CATEGORICAL_INK[1]},
            hovertemplate="Tendencia OLS: %{y:,.2f}<extra></extra>",
        )
    mann_kendall = mann_kendall_test(y_numeric.to_numpy())
    if mann_kendall is not None:
        base = float(np.nanmedian(y_numeric.to_numpy() - mann_kendall.sen_slope * x_numeric.to_numpy()))
        sen_values = base + mann_kendall.sen_slope * x_numeric.to_numpy()
        figure.add_scatter(
            x=x_numeric,
            y=sen_values,
            mode="lines",
            name=f"Sen ({mann_kendall.sen_slope:+.2f}/ano; p={mann_kendall.p_value:.3f})",
            line={"width": 2, "dash": "dot", "color": CATEGORICAL_INK[2]},
            hovertemplate="Pendiente de Sen: %{y:,.2f}<extra></extra>",
        )
    figure.update_xaxes(title_text="Ano", dtick=1)
    figure.update_yaxes(title_text=y_label or _label(value))
    return _finish(figure, title, height=450, showlegend=len(figure.data) > 1)


def histogram_distribution(
    frame: pd.DataFrame,
    *,
    variable: str,
    title: str,
    thresholds: Iterable[float] = (),
    bins: int = 60,
    show_kde: bool = True,
    category: str | None = None,
) -> go.Figure:
    """Muestra la distribucion de una variable con KDE y umbrales opcionales."""
    if frame.empty or variable not in frame:
        return _empty_figure(title)
    columns = [variable] + ([category] if category and category in frame else [])
    data = frame.loc[:, columns].copy()
    data[variable] = pd.to_numeric(data[variable], errors="coerce")
    data = data.dropna(subset=[variable])
    if data.empty:
        return _empty_figure(title)
    if category and category in data:
        data[category] = _collapse_categories(data[category], len(CATEGORICAL_FILL))
        groups = _category_order(data[category])
    else:
        groups = [""]
    figure = go.Figure()
    for index, group in enumerate(groups):
        values = data[variable] if not category else data.loc[data[category].astype(str) == group, variable]
        figure.add_histogram(
            x=values,
            nbinsx=bins,
            histnorm="probability density" if show_kde else None,
            opacity=0.64 if len(groups) > 1 else 0.82,
            name=group or _label(variable),
            marker_color=CATEGORICAL_FILL[index],
            hovertemplate=f"{_label(variable)}: %{{x:,.2f}}<br>Densidad/frecuencia: %{{y:,.4f}}<extra></extra>",
        )
        if show_kde and len(values) > 2 and float(values.std()) > 0:
            sample_values = values
            if len(values) > 20_000:
                sample_values = values.iloc[_sample_positions(len(values), 20_000)]
            try:
                kernel = stats.gaussian_kde(sample_values.to_numpy(dtype="float64"))
                grid = np.linspace(float(values.min()), float(values.max()), 240)
                figure.add_scatter(
                    x=grid,
                    y=kernel(grid),
                    mode="lines",
                    name=f"KDE{f' · {group}' if group else ''}",
                    line={"width": 2, "color": CATEGORICAL_INK[index]},
                    hovertemplate=f"{_label(variable)}: %{{x:,.2f}}<br>Densidad KDE: %{{y:,.4f}}<extra></extra>",
                )
            except (np.linalg.LinAlgError, ValueError):
                continue
    for index, threshold in enumerate(thresholds):
        figure.add_vline(
            x=float(threshold),
            line={"width": 1.5, "dash": "dash", "color": CATEGORICAL_INK[(index + 2) % len(CATEGORICAL_INK)]},
            annotation_text=f"Umbral {threshold:g}",
            annotation_position="top",
        )
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title_text=_label(variable))
    figure.update_yaxes(title_text="Densidad" if show_kde else "Frecuencia")
    return _finish(figure, title, height=430, showlegend=len(groups) > 1 or show_kde)


def box_by_category(
    frame: pd.DataFrame,
    *,
    variable: str,
    category: str,
    title: str,
    order: Sequence[Any] | None = None,
    show_points: bool = False,
) -> go.Figure:
    """Compara distribuciones mediante cajas por categoria."""
    if frame.empty or variable not in frame or category not in frame:
        return _empty_figure(title)
    data = frame.loc[:, [variable, category]].dropna().copy()
    groups = _category_order(data[category], order)
    figure = go.Figure()
    for group in groups:
        values = pd.to_numeric(data.loc[data[category].astype(str) == group, variable], errors="coerce").dropna()
        figure.add_box(
            y=values,
            name=group,
            boxpoints="outliers" if show_points else False,
            marker_color=CATEGORICAL_INK[0],
            line_color=CATEGORICAL_INK[0],
            fillcolor=hex_to_rgba(CATEGORICAL_FILL[0], 0.38),
            hovertemplate=f"<b>{group}</b><br>{_label(variable)}: %{{y:,.2f}}<extra></extra>",
            showlegend=False,
        )
    figure.update_xaxes(title_text=_label(category))
    figure.update_yaxes(title_text=_label(variable))
    return _finish(figure, title, height=450, showlegend=False)


def violin_by_category(
    frame: pd.DataFrame,
    *,
    variable: str,
    category: str,
    title: str,
    order: Sequence[Any] | None = None,
) -> go.Figure:
    """Compara densidades mediante violines por categoria."""
    if frame.empty or variable not in frame or category not in frame:
        return _empty_figure(title)
    data = frame.loc[:, [variable, category]].dropna().copy()
    groups = _category_order(data[category], order)
    figure = go.Figure()
    for group in groups:
        values = pd.to_numeric(data.loc[data[category].astype(str) == group, variable], errors="coerce").dropna()
        figure.add_violin(
            y=values,
            name=group,
            box_visible=True,
            meanline_visible=True,
            points=False,
            line_color=CATEGORICAL_INK[0],
            fillcolor=hex_to_rgba(CATEGORICAL_FILL[0], 0.46),
            hovertemplate=f"<b>{group}</b><br>{_label(variable)}: %{{y:,.2f}}<extra></extra>",
            showlegend=False,
        )
    figure.update_xaxes(title_text=_label(category))
    figure.update_yaxes(title_text=_label(variable))
    return _finish(figure, title, height=470, showlegend=False)


def scatter_relationship(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    size: str | None = None,
    show_regression: bool = True,
    sample: int | None = None,
    hover_extra: Sequence[str] | None = None,
) -> go.Figure:
    """Explora la relacion entre dos magnitudes con muestreo reproducible."""
    required = [x, y] + ([color] if color else []) + ([size] if size else []) + list(hover_extra or [])
    required = [column for column in dict.fromkeys(required) if column in frame]
    if frame.empty or x not in frame or y not in frame:
        return _empty_figure(title)
    data = frame.loc[:, required].dropna(subset=[x, y]).copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[x, y])
    sampled, did_sample = _sample_frame(data, sample)
    shown_title = _subtitle_for_sample(title, did_sample, len(sampled), len(data))
    figure = go.Figure()
    extras = [column for column in (hover_extra or []) if column in sampled]
    custom_columns = extras
    customdata = sampled[custom_columns].astype(str).to_numpy() if custom_columns else None
    extra_template = "".join(
        f"<br>{_label(column)}: %{{customdata[{index}]}}" for index, column in enumerate(custom_columns)
    )
    sizes: np.ndarray | float = 9
    if size and size in sampled:
        raw_size = pd.to_numeric(sampled[size], errors="coerce").fillna(0).clip(lower=0)
        maximum = float(raw_size.max())
        sizes = 7 + 18 * np.sqrt(raw_size.to_numpy() / maximum) if maximum > 0 else 9

    is_numeric_color = bool(color and pd.api.types.is_numeric_dtype(sampled[color]))
    if color and not is_numeric_color:
        sampled[color] = _collapse_categories(sampled[color], len(CATEGORICAL_ALL_PAIRS))
        groups = _category_order(sampled[color])
        for index, group in enumerate(groups):
            mask = sampled[color].astype(str) == group
            group_custom = customdata[mask.to_numpy()] if customdata is not None else None
            group_sizes = sizes[mask.to_numpy()] if isinstance(sizes, np.ndarray) else sizes
            figure.add_scattergl(
                x=sampled.loc[mask, x],
                y=sampled.loc[mask, y],
                mode="markers",
                name=group,
                marker={"size": group_sizes, "color": CATEGORICAL_ALL_PAIRS[index], "opacity": 0.7, "line": {"width": 1, "color": PALETTE["surface"]}},
                customdata=group_custom,
                hovertemplate=f"{_label(x)}: %{{x:,.2f}}<br>{_label(y)}: %{{y:,.2f}}<br>{_label(color)}: {group}{extra_template}<extra></extra>",
            )
    else:
        marker: dict[str, Any] = {"size": sizes, "opacity": 0.68, "line": {"width": 1, "color": PALETTE["surface"]}}
        if color and color in sampled:
            marker.update({"color": sampled[color], "colorscale": list(SEQUENTIAL_BLUE), "colorbar": _colorbar(_label(color)), "showscale": True})
        else:
            marker["color"] = CATEGORICAL_ALL_PAIRS[0]
        figure.add_scattergl(
            x=sampled[x],
            y=sampled[y],
            mode="markers",
            name=_label(y),
            marker=marker,
            customdata=customdata,
            hovertemplate=f"{_label(x)}: %{{x:,.2f}}<br>{_label(y)}: %{{y:,.2f}}{extra_template}<extra></extra>",
        )
    # SciPy rechaza una regresion cuando todos los valores explicativos son
    # iguales. Esto ocurre de forma legitima al filtrar una sola provincia y
    # usar su altitud como eje x, por lo que el grafico debe seguir disponible
    # sin la recta en vez de propagar una excepcion a Streamlit.
    if show_regression and data[x].nunique(dropna=True) > 1:
        trend = fit_linear_trend(data[x].to_numpy(), data[y].to_numpy())
        if trend is not None:
            x_line = np.array([float(data[x].min()), float(data[x].max())])
            figure.add_scatter(
                x=x_line,
                y=trend.intercept + trend.slope * x_line,
                mode="lines",
                name=f"OLS (R²={trend.r_squared:.2f}; p={trend.p_value:.3f})",
                line={"width": 2, "dash": "dash", "color": PALETTE["ink_primary"]},
                hovertemplate="Ajuste OLS: %{y:,.2f}<extra></extra>",
            )
    figure.update_xaxes(title_text=_label(x))
    figure.update_yaxes(title_text=_label(y))
    return _finish(figure, shown_title, height=470, showlegend=len(figure.data) > 1)


def heatmap_calendar(
    matrix: pd.DataFrame,
    *,
    title: str,
    colorbar_title: str = "%",
    value_format: str = ".0f",
) -> go.Figure:
    """Representa una matriz calendario con escala continua."""
    if matrix is None or matrix.empty:
        return _empty_figure(title)
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    texttemplate = f"%{{z:{value_format}}}" if numeric.size <= 220 else None
    figure = go.Figure(
        go.Heatmap(
            z=numeric.to_numpy(),
            x=[str(column) for column in numeric.columns],
            y=numeric.index.astype(str),
            colorscale=list(SEQUENTIAL_BLUE),
            colorbar=_colorbar(colorbar_title),
            texttemplate=texttemplate,
            hovertemplate=f"Fila: %{{y}}<br>Columna: %{{x}}<br>{colorbar_title}: %{{z:{value_format}}}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    figure.update_xaxes(title_text="Mes")
    figure.update_yaxes(title_text=matrix.index.name or "Categoria")
    return _finish(figure, title, height=max(420, 27 * len(matrix) + 170), showlegend=False)


def heatmap_correlation(
    corr: pd.DataFrame,
    pvalues: pd.DataFrame | None = None,
    *,
    title: str,
    alpha: float = 0.05,
) -> go.Figure:
    """Muestra correlaciones centradas en cero y marca resultados no significativos."""
    if corr is None or corr.empty:
        return _empty_figure(title)
    values = corr.apply(pd.to_numeric, errors="coerce")
    labels = [_label(column, str(column)) for column in values.columns]
    p_matrix = pvalues.reindex(index=corr.index, columns=corr.columns) if pvalues is not None else None
    text = np.empty(values.shape, dtype=object)
    custom = np.full(values.shape, np.nan)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            coefficient = values.iat[row, column]
            if pd.isna(coefficient):
                text[row, column] = ""
                continue
            p_value = float(p_matrix.iat[row, column]) if p_matrix is not None else np.nan
            custom[row, column] = p_value
            marker = " n.s." if p_matrix is not None and p_value >= alpha else ""
            text[row, column] = f"{coefficient:.2f}{marker}"
    hover = "<b>%{y} × %{x}</b><br>r: %{z:.3f}"
    if p_matrix is not None:
        hover += "<br>p: %{customdata:.4f}"
    hover += "<extra></extra>"
    figure = go.Figure(
        go.Heatmap(
            z=values.to_numpy(),
            x=labels,
            y=[_label(index, str(index)) for index in values.index],
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=list(DIVERGING_THERMAL),
            colorbar=_colorbar("Coeficiente r"),
            text=text,
            texttemplate="%{text}",
            customdata=custom,
            hovertemplate=hover,
            xgap=2,
            ygap=2,
        )
    )
    figure.update_xaxes(tickangle=-35)
    return _finish(figure, title, height=max(500, 38 * len(values) + 170), showlegend=False)


def map_provinces(
    summary: pd.DataFrame,
    *,
    color: str,
    size: str,
    title: str,
    color_label: str | None = None,
) -> go.Figure:
    """Situa las provincias de Puno en un mapa de burbujas."""
    required = {"latitud", "longitud", color, size}
    if summary.empty or not required.issubset(summary.columns):
        return _empty_figure(title)
    data = summary.dropna(subset=list(required)).copy()
    hover_name = "provincia" if "provincia" in data else None
    hover_data = {
        column: ":,.2f"
        for column in (color, size, "altitud", "tmin_media", "tasa_helada")
        if column in data
    }
    figure = px.scatter_map(
        data,
        lat="latitud",
        lon="longitud",
        color=color,
        size=size,
        hover_name=hover_name,
        hover_data=hover_data,
        color_continuous_scale=list(SEQUENTIAL_ORANGE),
        size_max=34,
        zoom=6.6,
        center={"lat": -15.3, "lon": -69.8},
        map_style="carto-positron",
    )
    for trace in figure.data:
        trace.hovertemplate = (
            "<b>%{hovertext}</b><br>Latitud: %{lat:.3f}<br>Longitud: %{lon:.3f}"
            f"<br>{color_label or _label(color)}: %{{marker.color:,.2f}}"
            f"<br>{_label(size)}: %{{marker.size:,.2f}}<extra></extra>"
        )
    figure.update_coloraxes(colorbar=_colorbar(color_label or _label(color)))
    figure.update_layout(margin={"l": 0, "r": 0, "t": 56, "b": 0})
    return _finish(figure, title, height=560, showlegend=False)


def treemap_territory(summary: pd.DataFrame, *, value: str, title: str) -> go.Figure:
    """Descompone una magnitud por cuenca, piso ecologico y provincia."""
    if summary.empty or value not in summary:
        return _empty_figure(title)
    hierarchy = [column for column in ("cuenca", "piso_ecologico", "provincia") if column in summary]
    if not hierarchy:
        return _empty_figure(title, "No hay categorias territoriales disponibles.")
    data = summary.dropna(subset=[value]).copy()
    data[value] = pd.to_numeric(data[value], errors="coerce").clip(lower=0)
    data = data.dropna(subset=[value])
    if data.empty or float(data[value].sum()) <= 0:
        return _empty_figure(title)
    figure = px.treemap(
        data,
        path=[px.Constant("Region Puno"), *hierarchy],
        values=value,
        color=value,
        color_continuous_scale=list(SEQUENTIAL_ORANGE),
    )
    figure.update_traces(
        marker={"cornerradius": 4},
        hovertemplate=f"<b>%{{label}}</b><br>{_label(value)}: %{{value:,.2f}}<br>Participacion: %{{percentRoot:.1%}}<extra></extra>",
    )
    figure.update_coloraxes(colorbar=_colorbar(_label(value)))
    return _finish(figure, title, height=520, showlegend=False)


def pca_scree(pca: Any, *, title: str) -> go.Figure:
    """Grafica varianza individual y acumulada del PCA en paneles separados."""
    if pca is None or not hasattr(pca, "explained_variance"):
        return _empty_figure(title)
    explained = np.asarray(pca.explained_variance, dtype=float) * 100
    cumulative = np.asarray(pca.cumulative_variance, dtype=float) * 100
    components = [f"CP{index + 1}" for index in range(len(explained))]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13)
    figure.add_bar(
        x=components,
        y=explained,
        name="Varianza individual",
        marker_color=CATEGORICAL_FILL[0],
        text=[f"{value:.1f}%" for value in explained],
        hovertemplate="Componente: %{x}<br>Varianza: %{y:.2f}%<extra></extra>",
        row=1,
        col=1,
    )
    figure.add_scatter(
        x=components,
        y=cumulative,
        mode="lines+markers",
        name="Varianza acumulada",
        line={"width": 2, "color": CATEGORICAL_INK[1]},
        marker={"size": 9, "line": {"width": 1.5, "color": PALETTE["surface"]}},
        hovertemplate="Componente: %{x}<br>Acumulada: %{y:.2f}%<extra></extra>",
        row=2,
        col=1,
    )
    figure.add_hline(y=90, line_dash="dot", line_color=PALETTE["ink_muted"], row=2, col=1)
    figure.update_yaxes(title_text="Varianza (%)", row=1, col=1)
    figure.update_yaxes(title_text="Acumulada (%)", range=[0, 105], row=2, col=1)
    figure.update_xaxes(title_text="Componente principal", row=2, col=1)
    return _finish(figure, title, height=570, showlegend=True)


def pca_loadings_heatmap(pca: Any, *, title: str) -> go.Figure:
    """Muestra el peso de cada variable en las componentes principales."""
    if pca is None or not hasattr(pca, "loadings") or pca.loadings.empty:
        return _empty_figure(title)
    loadings = pca.loadings.copy()
    figure = go.Figure(
        go.Heatmap(
            z=loadings.to_numpy(),
            x=loadings.columns.astype(str),
            y=[_label(index, str(index)) for index in loadings.index],
            zmid=0,
            colorscale=list(DIVERGING_THERMAL),
            colorbar=_colorbar("Carga"),
            text=np.vectorize(lambda value: f"{value:.2f}")(loadings.to_numpy()),
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>Componente: %{x}<br>Carga: %{z:.3f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    return _finish(figure, title, height=max(480, 32 * len(loadings) + 160), showlegend=False)


def _pca_color_payload(values: np.ndarray | None) -> tuple[np.ndarray | None, list[str] | None]:
    if values is None:
        return None, None
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.95 and series.nunique(dropna=True) > 6:
        return numeric.to_numpy(), None
    collapsed = _collapse_categories(series, len(CATEGORICAL_ALL_PAIRS))
    categories = _category_order(collapsed)
    codes = collapsed.map({name: index for index, name in enumerate(categories)}).to_numpy()
    return codes, categories


def pca_biplot(
    pca: Any,
    *,
    color_values: Any = None,
    color_label: str | None = None,
    title: str = "",
    sample: int | None = None,
) -> go.Figure:
    """Proyecta observaciones y vectores de carga en las dos primeras CP."""
    if pca is None or np.asarray(getattr(pca, "scores", [])).ndim != 2 or pca.scores.shape[1] < 2:
        return _empty_figure(title)
    positions = _sample_positions(len(pca.scores), sample)
    scores = np.asarray(pca.scores)[positions]
    aligned = _series_values(color_values, pca.row_index, len(pca.scores))
    selected = aligned[positions] if aligned is not None else None
    payload, categories = _pca_color_payload(selected)
    figure = go.Figure()
    if categories is None:
        marker: dict[str, Any] = {"size": 8, "opacity": 0.58, "line": {"width": 1, "color": PALETTE["surface"]}}
        if payload is None:
            marker["color"] = CATEGORICAL_ALL_PAIRS[0]
        else:
            marker.update({"color": payload, "colorscale": list(SEQUENTIAL_BLUE), "colorbar": _colorbar(color_label or "Valor")})
        figure.add_scattergl(
            x=scores[:, 0],
            y=scores[:, 1],
            mode="markers",
            marker=marker,
            name="Observaciones",
            hovertemplate="CP1: %{x:.3f}<br>CP2: %{y:.3f}<extra></extra>",
        )
    else:
        for index, name in enumerate(categories):
            mask = payload == index
            figure.add_scattergl(
                x=scores[mask, 0],
                y=scores[mask, 1],
                mode="markers",
                marker={"size": 8, "color": CATEGORICAL_ALL_PAIRS[index], "opacity": 0.62, "line": {"width": 1, "color": PALETTE["surface"]}},
                name=name,
                hovertemplate=f"{color_label or 'Categoria'}: {name}<br>CP1: %{{x:.3f}}<br>CP2: %{{y:.3f}}<extra></extra>",
            )
    scale = 0.78 * min(float(np.nanmax(np.abs(scores[:, 0]))), float(np.nanmax(np.abs(scores[:, 1]))))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    loadings = pca.loadings.iloc[:, :2]
    for feature, row in loadings.iterrows():
        end_x, end_y = float(row.iloc[0]) * scale, float(row.iloc[1]) * scale
        figure.add_annotation(
            x=end_x,
            y=end_y,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=_label(str(feature), str(feature)),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=PALETTE["brand_ink"],
            font={"size": 10, "color": PALETTE["brand_ink"]},
        )
    shown_title = _subtitle_for_sample(title, len(positions) < len(pca.scores), len(positions), len(pca.scores))
    figure.update_xaxes(title_text=pca.component_label(0))
    figure.update_yaxes(title_text=pca.component_label(1))
    return _finish(figure, shown_title, height=560, showlegend=categories is not None and len(categories) > 1)


def pca_scatter_3d(
    pca: Any,
    *,
    color_values: Any = None,
    color_label: str | None = None,
    title: str = "",
    sample: int | None = None,
) -> go.Figure:
    """Proyecta observaciones en las tres primeras componentes."""
    if pca is None or np.asarray(getattr(pca, "scores", [])).ndim != 2 or pca.scores.shape[1] < 3:
        return _empty_figure(title, "El PCA necesita al menos tres componentes.")
    positions = _sample_positions(len(pca.scores), sample)
    scores = np.asarray(pca.scores)[positions]
    aligned = _series_values(color_values, pca.row_index, len(pca.scores))
    selected = aligned[positions] if aligned is not None else None
    payload, categories = _pca_color_payload(selected)
    figure = go.Figure()
    if categories is None:
        marker: dict[str, Any] = {"size": 4.5, "opacity": 0.62}
        if payload is None:
            marker["color"] = CATEGORICAL_ALL_PAIRS[0]
        else:
            marker.update({"color": payload, "colorscale": list(SEQUENTIAL_BLUE), "colorbar": _colorbar(color_label or "Valor")})
        figure.add_scatter3d(
            x=scores[:, 0], y=scores[:, 1], z=scores[:, 2], mode="markers", marker=marker,
            name="Observaciones", hovertemplate="CP1: %{x:.3f}<br>CP2: %{y:.3f}<br>CP3: %{z:.3f}<extra></extra>",
        )
    else:
        for index, name in enumerate(categories):
            mask = payload == index
            figure.add_scatter3d(
                x=scores[mask, 0], y=scores[mask, 1], z=scores[mask, 2], mode="markers",
                marker={"size": 4.5, "color": CATEGORICAL_ALL_PAIRS[index], "opacity": 0.65},
                name=name,
                hovertemplate=f"{color_label or 'Categoria'}: {name}<br>CP1: %{{x:.3f}}<br>CP2: %{{y:.3f}}<br>CP3: %{{z:.3f}}<extra></extra>",
            )
    figure.update_layout(
        scene={
            "xaxis_title": pca.component_label(0),
            "yaxis_title": pca.component_label(1),
            "zaxis_title": pca.component_label(2),
            "bgcolor": PALETTE["surface"],
        }
    )
    shown_title = _subtitle_for_sample(title, len(positions) < len(pca.scores), len(positions), len(pca.scores))
    return _finish(figure, shown_title, height=610, showlegend=categories is not None and len(categories) > 1)


def cluster_scatter(
    pca: Any,
    labels: Sequence[int],
    *,
    title: str,
    cluster_names: Any = None,
    sample: int | None = None,
) -> go.Figure:
    """Muestra grupos en el plano de las dos primeras componentes."""
    if pca is None or pca.scores.shape[1] < 2 or len(labels) != len(pca.scores):
        return _empty_figure(title)
    positions = _sample_positions(len(labels), sample)
    scores = np.asarray(pca.scores)[positions]
    selected_labels = np.asarray(labels)[positions]
    unique = sorted(pd.unique(selected_labels).tolist())[: len(CATEGORICAL_ALL_PAIRS)]
    figure = go.Figure()
    for index, label in enumerate(unique):
        mask = selected_labels == label
        if isinstance(cluster_names, dict):
            name = str(cluster_names.get(label, cluster_names.get(int(label), f"Grupo {int(label) + 1}")))
        elif isinstance(cluster_names, Sequence) and not isinstance(cluster_names, str) and int(label) < len(cluster_names):
            name = str(cluster_names[int(label)])
        else:
            name = f"Grupo {int(label) + 1}"
        figure.add_scattergl(
            x=scores[mask, 0],
            y=scores[mask, 1],
            mode="markers",
            name=name,
            marker={"size": 8, "color": CATEGORICAL_ALL_PAIRS[index], "opacity": 0.68, "line": {"width": 1, "color": PALETTE["surface"]}},
            hovertemplate=f"<b>{name}</b><br>CP1: %{{x:.3f}}<br>CP2: %{{y:.3f}}<extra></extra>",
        )
    shown_title = _subtitle_for_sample(title, len(positions) < len(labels), len(positions), len(labels))
    figure.update_xaxes(title_text=pca.component_label(0))
    figure.update_yaxes(title_text=pca.component_label(1))
    return _finish(figure, shown_title, height=510, showlegend=len(unique) > 1)


def cluster_radar(
    profile: pd.DataFrame,
    *,
    features: Sequence[str],
    title: str,
    cluster_names: Any = None,
) -> go.Figure:
    """Compara perfiles de grupo en una escala 0-1 por variable."""
    available = [feature for feature in features if feature in profile]
    if profile.empty or len(available) < 3:
        return _empty_figure(title, "Se necesitan al menos tres variables de perfil.")
    data = profile.loc[:, available].apply(pd.to_numeric, errors="coerce")
    minima, ranges = data.min(), (data.max() - data.min()).replace(0, 1)
    normalized = (data - minima) / ranges
    categories = [_label(feature, feature) for feature in available]
    figure = go.Figure()
    for position, (row_index, row) in enumerate(normalized.head(len(CATEGORICAL_ALL_PAIRS)).iterrows()):
        group_value = profile.loc[row_index, "grupo"] if "grupo" in profile else position
        if isinstance(cluster_names, dict):
            name = str(cluster_names.get(group_value, f"Grupo {int(group_value) + 1}"))
        elif isinstance(cluster_names, Sequence) and not isinstance(cluster_names, str) and position < len(cluster_names):
            name = str(cluster_names[position])
        elif "etiqueta" in profile:
            name = str(profile.loc[row_index, "etiqueta"])
        else:
            name = f"Grupo {int(group_value) + 1}"
        theta = [*categories, categories[0]]
        radial = [*row.to_list(), float(row.iloc[0])]
        figure.add_scatterpolar(
            theta=theta,
            r=radial,
            mode="lines+markers",
            fill="toself",
            name=name,
            line={"width": 2, "color": CATEGORICAL_ALL_PAIRS[position]},
            marker={"size": 8, "line": {"width": 1.5, "color": PALETTE["surface"]}},
            fillcolor=hex_to_rgba(CATEGORICAL_ALL_PAIRS[position], 0.12),
            hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.2f}} (normalizado)<extra></extra>",
        )
    figure.update_layout(polar={"radialaxis": {"range": [0, 1], "tickformat": ".1f"}})
    return _finish(figure, title, height=560, showlegend=len(figure.data) > 1)


def elbow_silhouette(diagnostics: pd.DataFrame, *, title: str) -> go.Figure:
    """Muestra inercia y silueta en dos paneles apilados."""
    required = {"k", "inercia", "silueta"}
    if diagnostics.empty or not required.issubset(diagnostics.columns):
        return _empty_figure(title)
    data = diagnostics.sort_values("k")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13)
    figure.add_scatter(
        x=data["k"], y=data["inercia"], mode="lines+markers", name="Inercia",
        line={"width": 2, "color": CATEGORICAL_INK[0]},
        marker={"size": 9, "line": {"width": 1.5, "color": PALETTE["surface"]}},
        hovertemplate="k: %{x:.0f}<br>Inercia: %{y:,.2f}<extra></extra>", row=1, col=1,
    )
    figure.add_scatter(
        x=data["k"], y=data["silueta"], mode="lines+markers", name="Silueta",
        line={"width": 2, "color": CATEGORICAL_INK[1]},
        marker={"size": 9, "line": {"width": 1.5, "color": PALETTE["surface"]}},
        hovertemplate="k: %{x:.0f}<br>Silueta: %{y:.3f}<extra></extra>", row=2, col=1,
    )
    figure.update_yaxes(title_text="Inercia", row=1, col=1)
    figure.update_yaxes(title_text="Silueta", row=2, col=1)
    figure.update_xaxes(title_text="Numero de grupos (k)", dtick=1, row=2, col=1)
    return _finish(figure, title, height=560, showlegend=True)


def dendrogram_chart(hierarchical: Any, *, title: str) -> go.Figure:
    """Convierte un enlace jerarquico de SciPy en un dendrograma Plotly."""
    if hierarchical is None or not hasattr(hierarchical, "linkage_matrix"):
        return _empty_figure(title)
    result = dendrogram(
        hierarchical.linkage_matrix,
        labels=list(hierarchical.province_names),
        orientation="top",
        no_plot=True,
        color_threshold=None,
    )
    figure = go.Figure()
    for index, (x_values, y_values) in enumerate(zip(result["icoord"], result["dcoord"])):
        color = CATEGORICAL_INK[index % min(4, len(CATEGORICAL_INK))]
        figure.add_scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line={"width": 2, "color": color},
            showlegend=False,
            hovertemplate="Distancia de fusion: %{y:.3f}<extra></extra>",
        )
    tick_values = [5 + 10 * index for index in range(len(result["ivl"]))]
    figure.update_xaxes(title_text="Provincia", tickmode="array", tickvals=tick_values, ticktext=result["ivl"], tickangle=-35)
    figure.update_yaxes(title_text="Distancia de Ward")
    return _finish(figure, title, height=530, showlegend=False)


def parallel_coordinates(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    color: str,
    title: str,
    sample: int | None = None,
) -> go.Figure:
    """Compara perfiles multivariantes mediante coordenadas paralelas."""
    available = [feature for feature in features if feature in frame]
    if frame.empty or len(available) < 2 or color not in frame:
        return _empty_figure(title)
    data = frame.loc[:, [*available, color]].dropna().copy()
    data, did_sample = _sample_frame(data, sample)
    color_values = pd.to_numeric(data[color], errors="coerce")
    tickvals = ticktext = None
    if color_values.isna().any():
        categories = _category_order(_collapse_categories(data[color], 4))
        mapping = {name: index for index, name in enumerate(categories)}
        color_values = _collapse_categories(data[color], 4).map(mapping).astype(float)
        tickvals, ticktext = list(mapping.values()), list(mapping.keys())
    dimensions = [
        {"label": _label(feature, feature), "values": pd.to_numeric(data[feature], errors="coerce")}
        for feature in available
    ]
    colorbar = _colorbar(_label(color, color))
    if tickvals is not None:
        colorbar.update({"tickvals": tickvals, "ticktext": ticktext})
    figure = go.Figure(
        go.Parcoords(
            line={"color": color_values, "colorscale": list(SEQUENTIAL_BLUE), "showscale": True, "colorbar": colorbar},
            dimensions=dimensions,
            labelfont={"size": 11, "color": PALETTE["ink_secondary"]},
            tickfont={"size": 10, "color": PALETTE["ink_muted"]},
        )
    )
    shown_title = _subtitle_for_sample(title, did_sample, len(data), len(frame))
    return _finish(figure, shown_title, height=540, showlegend=False)


def anomaly_timeline(
    frame: pd.DataFrame,
    flags: Sequence[bool] | pd.Series,
    *,
    variable: str,
    title: str,
) -> go.Figure:
    """Superpone eventos anomalos sobre la evolucion diaria de una variable."""
    if frame.empty or variable not in frame or "fecha" not in frame or len(flags) != len(frame):
        return _empty_figure(title)
    data = frame.loc[:, ["fecha", variable] + (["provincia"] if "provincia" in frame else [])].copy()
    if isinstance(flags, pd.Series):
        data["anomalia"] = flags.reindex(frame.index, fill_value=False).to_numpy(dtype=bool)
    else:
        data["anomalia"] = np.asarray(flags, dtype=bool)
    data[variable] = pd.to_numeric(data[variable], errors="coerce")
    data = data.dropna(subset=["fecha", variable])
    daily = data.groupby("fecha", as_index=False)[variable].mean()
    anomalies = data[data["anomalia"]]
    figure = go.Figure()
    figure.add_scatter(
        x=daily["fecha"], y=daily[variable], mode="lines", name="Promedio diario",
        line={"width": 2, "color": CATEGORICAL_INK[1]},
        hovertemplate=f"Fecha: %{{x|%d/%m/%Y}}<br>{_label(variable)}: %{{y:,.2f}}<extra></extra>",
    )
    hover_columns = ["provincia"] if "provincia" in anomalies else []
    figure.add_scattergl(
        x=anomalies["fecha"], y=anomalies[variable], mode="markers", name="Anomalia",
        marker={"size": 10, "color": CATEGORICAL_ALL_PAIRS[0], "symbol": "diamond", "line": {"width": 1.5, "color": PALETTE["surface"]}},
        customdata=anomalies[hover_columns].astype(str).to_numpy() if hover_columns else None,
        hovertemplate=(
            f"<b>Anomalia</b><br>Fecha: %{{x|%d/%m/%Y}}<br>{_label(variable)}: %{{y:,.2f}}"
            + ("<br>Provincia: %{customdata[0]}" if hover_columns else "")
            + "<extra></extra>"
        ),
    )
    figure.update_xaxes(title_text="Fecha")
    figure.update_yaxes(title_text=_label(variable))
    return _finish(figure, title, height=470, showlegend=True)


def importance_bar(importances: pd.DataFrame, *, title: str, top_n: int = 12) -> go.Figure:
    """Ordena la importancia por permutacion de un clasificador."""
    if importances is None or importances.empty or "importancia" not in importances:
        return _empty_figure(title)
    label_column = "etiqueta" if "etiqueta" in importances else "variable"
    if label_column not in importances:
        return _empty_figure(title)
    data = importances.nlargest(top_n, "importancia").sort_values("importancia")
    errors = data["desviacion"] if "desviacion" in data else None
    figure = go.Figure(
        go.Bar(
            x=data["importancia"], y=data[label_column], orientation="h",
            marker_color=CATEGORICAL_FILL[0], error_x={"type": "data", "array": errors, "visible": errors is not None},
            text=data["importancia"].map(lambda value: f"{value:.3f}"),
            hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Disminucion de exactitud por permutacion")
    figure.update_yaxes(title_text="Variable")
    return _finish(figure, title, height=max(410, 31 * len(data) + 150), showlegend=False)


def roc_curve_chart(classifier: Any, *, title: str) -> go.Figure:
    """Grafica la curva ROC y su referencia aleatoria."""
    if classifier is None or not hasattr(classifier, "roc_curve_points"):
        return _empty_figure(title)
    fpr, tpr = classifier.roc_curve_points
    figure = go.Figure()
    figure.add_scatter(
        x=fpr, y=tpr, mode="lines", name=f"Modelo (AUC={classifier.roc_auc:.3f})",
        line={"width": 2, "color": CATEGORICAL_INK[0]},
        hovertemplate="Falsos positivos: %{x:.3f}<br>Verdaderos positivos: %{y:.3f}<extra></extra>",
    )
    figure.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Azar",
        line={"width": 2, "dash": "dot", "color": PALETTE["ink_faint"]},
        hovertemplate="Referencia aleatoria<extra></extra>",
    )
    figure.update_xaxes(title_text="Tasa de falsos positivos", range=[0, 1])
    figure.update_yaxes(title_text="Tasa de verdaderos positivos", range=[0, 1])
    return _finish(figure, title, height=450, showlegend=True)


def precision_recall_chart(classifier: Any, *, title: str) -> go.Figure:
    """Grafica la curva precision-recall y la prevalencia de referencia."""
    if classifier is None or not hasattr(classifier, "pr_curve_points"):
        return _empty_figure(title)
    recall, precision = classifier.pr_curve_points
    figure = go.Figure()
    figure.add_scatter(
        x=recall, y=precision, mode="lines", name=f"Modelo (AP={classifier.average_precision:.3f})",
        line={"width": 2, "color": CATEGORICAL_INK[1]},
        hovertemplate="Exhaustividad: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
    )
    figure.add_scatter(
        x=[0, 1], y=[classifier.positive_rate, classifier.positive_rate], mode="lines",
        name=f"Prevalencia ({classifier.positive_rate:.1%})",
        line={"width": 2, "dash": "dot", "color": PALETTE["ink_faint"]},
        hovertemplate="Prevalencia: %{y:.3f}<extra></extra>",
    )
    figure.update_xaxes(title_text="Exhaustividad (recall)", range=[0, 1])
    figure.update_yaxes(title_text="Precision", range=[0, 1])
    return _finish(figure, title, height=450, showlegend=True)


def confusion_heatmap(classifier: Any, *, title: str) -> go.Figure:
    """Representa la matriz de confusion del clasificador."""
    if classifier is None or not hasattr(classifier, "confusion"):
        return _empty_figure(title)
    matrix = np.asarray(classifier.confusion)
    if matrix.shape != (2, 2):
        return _empty_figure(title)
    labels = ["Sin helada", "Con helada"]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=list(SEQUENTIAL_BLUE),
            colorbar=_colorbar("Casos"),
            text=matrix.astype(str),
            texttemplate="%{text}",
            hovertemplate="Real: %{y}<br>Predicha: %{x}<br>Casos: %{z:,.0f}<extra></extra>",
            xgap=3,
            ygap=3,
        )
    )
    figure.update_xaxes(title_text="Clase predicha")
    figure.update_yaxes(title_text="Clase real", autorange="reversed")
    return _finish(figure, title, height=430, showlegend=False)


def source_agreement_scatter(frame: pd.DataFrame, *, title: str, sample: int | None = None) -> go.Figure:
    """Compara temperatura minima ERA5-Land y MERRA-2 contra la recta 1:1."""
    primary, secondary = "temperatura_minima", "temperatura_minima_merra2"
    if frame.empty or primary not in frame or secondary not in frame:
        return _empty_figure(title)
    data = frame.loc[:, [primary, secondary] + (["provincia"] if "provincia" in frame else [])].dropna()
    sampled, did_sample = _sample_frame(data, sample)
    customdata = sampled[["provincia"]].astype(str).to_numpy() if "provincia" in sampled else None
    figure = go.Figure()
    figure.add_scattergl(
        x=sampled[primary], y=sampled[secondary], mode="markers", name="Pares diarios",
        marker={"size": 8, "color": CATEGORICAL_ALL_PAIRS[1], "opacity": 0.55, "line": {"width": 1, "color": PALETTE["surface"]}},
        customdata=customdata,
        hovertemplate=(
            "ERA5-Land (°C): %{x:.2f}<br>MERRA-2 (°C): %{y:.2f}"
            + ("<br>Provincia: %{customdata[0]}" if customdata is not None else "")
            + "<extra></extra>"
        ),
    )
    low = float(min(data[primary].min(), data[secondary].min()))
    high = float(max(data[primary].max(), data[secondary].max()))
    figure.add_scatter(
        x=[low, high], y=[low, high], mode="lines", name="Concordancia perfecta (1:1)",
        line={"width": 2, "dash": "dot", "color": PALETTE["ink_primary"]},
        hovertemplate="Referencia 1:1<extra></extra>",
    )
    trend = (
        fit_linear_trend(data[primary].to_numpy(), data[secondary].to_numpy())
        if data[primary].nunique(dropna=True) > 1
        else None
    )
    if trend is not None:
        x_line = np.array([low, high])
        figure.add_scatter(
            x=x_line, y=trend.intercept + trend.slope * x_line, mode="lines",
            name=f"OLS (R²={trend.r_squared:.2f})", line={"width": 2, "dash": "dash", "color": CATEGORICAL_INK[0]},
            hovertemplate="Ajuste OLS: %{y:.2f} °C<extra></extra>",
        )
    shown_title = _subtitle_for_sample(title, did_sample, len(sampled), len(data))
    figure.update_xaxes(title_text="Temperatura minima ERA5-Land (°C)", scaleanchor="y", scaleratio=1)
    figure.update_yaxes(title_text="Temperatura minima MERRA-2 (°C)")
    return _finish(figure, shown_title, height=510, showlegend=True)


def ecdf_chart(
    frame: pd.DataFrame,
    *,
    variable: str,
    title: str,
    category: str | None = None,
) -> go.Figure:
    """Grafica la funcion de distribucion acumulada empirica."""
    if frame.empty or variable not in frame:
        return _empty_figure(title)
    data = frame.loc[:, [variable] + ([category] if category and category in frame else [])].dropna(subset=[variable]).copy()
    if category and category in data:
        data[category] = _collapse_categories(data[category], len(CATEGORICAL_INK))
        groups = _category_order(data[category])
    else:
        groups = [""]
    figure = go.Figure()
    for index, group in enumerate(groups):
        values = data[variable] if not category else data.loc[data[category].astype(str) == group, variable]
        values = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy())
        if not len(values):
            continue
        probabilities = np.arange(1, len(values) + 1) / len(values)
        figure.add_scatter(
            x=values, y=probabilities, mode="lines", name=group or _label(variable),
            line={"width": 2, "color": CATEGORICAL_INK[index]},
            hovertemplate=f"{_label(variable)}: %{{x:,.2f}}<br>Proporcion acumulada: %{{y:.1%}}<extra></extra>",
        )
    figure.update_xaxes(title_text=_label(variable))
    figure.update_yaxes(title_text="Proporcion acumulada", tickformat=".0%", range=[0, 1.01])
    return _finish(figure, title, height=430, showlegend=len(groups) > 1)


def gauge_risk(value: float, *, title: str) -> go.Figure:
    """Muestra el indice de riesgo en una escala ordinal de 0 a 100."""
    numeric = float(np.clip(value if np.isfinite(value) else 0, 0, 100))
    levels = list(RISK_LEVEL_COLORS.items())
    width = 100 / len(levels)
    steps = [
        {"range": [index * width, (index + 1) * width], "color": color}
        for index, (_, color) in enumerate(levels)
    ]
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=numeric,
            number={"valueformat": ".1f", "font": {"size": 36, "color": PALETTE["ink_primary"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PALETTE["axis"]},
                "bar": {"color": PALETTE["ink_primary"], "thickness": 0.24},
                "bgcolor": PALETTE["surface"],
                "borderwidth": 0,
                "steps": steps,
                "threshold": {"line": {"color": PALETTE["brand_ink"], "width": 3}, "thickness": 0.75, "value": numeric},
            },
        )
    )
    figure.add_annotation(
        x=0.5,
        y=0.08,
        text=" · ".join(name for name, _ in levels),
        showarrow=False,
        font={"size": 10, "color": PALETTE["ink_muted"]},
    )
    return _finish(figure, title, height=360, showlegend=False)


__all__ = [
    "render",
    "bar_ranking",
    "bar_grouped",
    "bar_stacked_intensity",
    "line_series",
    "line_monthly_cycle",
    "line_annual_with_trend",
    "histogram_distribution",
    "box_by_category",
    "violin_by_category",
    "scatter_relationship",
    "heatmap_calendar",
    "heatmap_correlation",
    "map_provinces",
    "treemap_territory",
    "pca_scree",
    "pca_loadings_heatmap",
    "pca_biplot",
    "pca_scatter_3d",
    "cluster_scatter",
    "cluster_radar",
    "elbow_silhouette",
    "dendrogram_chart",
    "parallel_coordinates",
    "anomaly_timeline",
    "importance_bar",
    "roc_curve_chart",
    "precision_recall_chart",
    "confusion_heatmap",
    "source_agreement_scatter",
    "ecdf_chart",
    "gauge_risk",
]
