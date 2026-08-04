"""Panel de filtros global, compartido por todas las páginas.

El estado de los controles vive en ``st.session_state`` con claves estables, lo
que produce el comportamiento que la consigna exige: la selección hecha en una
página sigue vigente al navegar a otra, y los indicadores y gráficos de la nueva
página se recalculan sobre el mismo subconjunto. Es el filtrado cruzado real, no
una simulación por página.

Se ofrecen ocho filtros —la consigna exige un mínimo de tres— cubriendo las seis
categorías que enumera: rango de fechas, categoría, ubicación, variable de
análisis, rango de valores y selección múltiple.
"""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import (
    COURSE,
    FROST_THRESHOLD_LABELS,
    NUMERIC_ANALYSIS_VARIABLES,
    PROJECT_PATHS,
)
from utils.filters import FilterState, apply_filters, build_default_state, filter_impact
from utils.formatting import format_integer, format_percent, variable_label

#: Prefijo de las claves de estado. Evita colisiones con claves de widgets que
#: Streamlit genere internamente.
_KEY = "pn_filter"


def _render_brand() -> None:
    """Dibuja la marca del proyecto en la cabecera de la barra lateral."""
    logo_path = PROJECT_PATHS.logos / "logo_unap.png"
    logo_markup = ""
    if logo_path.exists():
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        logo_markup = f'<img src="data:image/png;base64,{encoded}" alt="Escudo UNAP">'

    st.sidebar.markdown(
        f"""
        <div class="pn-sidebar-brand">
          {logo_markup}
          <div class="pn-sidebar-brand__text">
            <span class="pn-sidebar-brand__title">Heladas · Puno</span>
            <span class="pn-sidebar-brand__subtitle">{COURSE['course_code']} · {COURSE['semester']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _group_label(text: str) -> None:
    """Dibuja el rótulo de un grupo de filtros.

    Args:
        text: Nombre del grupo.
    """
    st.sidebar.markdown(f'<div class="pn-filter-group">{text}</div>', unsafe_allow_html=True)


def render_sidebar(frame: pd.DataFrame) -> tuple[FilterState, pd.DataFrame]:
    """Dibuja el panel de filtros y devuelve el estado y el subconjunto filtrado.

    Args:
        frame: Dataset analítico completo.

    Returns:
        Tupla ``(estado_de_filtros, dataframe_filtrado)``.
    """
    defaults = build_default_state(frame)
    _render_brand()

    minimum_date: date = frame["fecha"].min().date()
    maximum_date: date = frame["fecha"].max().date()

    # --- Periodo -----------------------------------------------------------
    _group_label("Periodo de análisis")

    preset = st.sidebar.selectbox(
        "Atajo temporal",
        options=[
            "Todo el periodo",
            "Últimos 3 años",
            "Últimos 5 años",
            "Última campaña agrícola",
            "Personalizado",
        ],
        key=f"{_KEY}_preset",
        help=(
            "Los atajos recalculan el rango de fechas respecto al último día "
            "disponible en el conjunto de datos."
        ),
    )

    resolved_range = _resolve_preset(preset, minimum_date, maximum_date)
    if resolved_range is None:
        selected = st.sidebar.date_input(
            "Rango de fechas",
            value=st.session_state.get(f"{_KEY}_dates", (minimum_date, maximum_date)),
            min_value=minimum_date,
            max_value=maximum_date,
            key=f"{_KEY}_dates",
            format="DD/MM/YYYY",
        )
        # Mientras el usuario elige la segunda fecha, el control devuelve una
        # sola: se conserva el extremo superior para no romper el filtrado.
        if isinstance(selected, (tuple, list)):
            date_range = (
                (selected[0], selected[1]) if len(selected) == 2 else (selected[0], maximum_date)
            )
        else:
            date_range = (selected, maximum_date)
    else:
        date_range = resolved_range
        st.sidebar.caption(
            f"Rango aplicado: {date_range[0]:%d/%m/%Y} – {date_range[1]:%d/%m/%Y}"
        )

    # --- Territorio --------------------------------------------------------
    _group_label("Territorio")

    provinces = st.sidebar.multiselect(
        "Provincias",
        options=sorted(frame["provincia"].cat.categories.tolist()),
        default=[],
        key=f"{_KEY}_provinces",
        placeholder="Todas las provincias",
        help="Sin selección se incluyen las trece provincias de la región.",
    )

    tiers = st.sidebar.multiselect(
        "Piso ecológico",
        options=[
            tier
            for tier in frame["piso_ecologico"].cat.categories.tolist()
            if tier in set(frame["piso_ecologico"].dropna().unique())
        ],
        default=[],
        key=f"{_KEY}_tiers",
        placeholder="Todos los pisos",
        help="Clasificación altitudinal según Pulgar Vidal (1981).",
    )

    altitude_bounds = (float(frame["altitud"].min()), float(frame["altitud"].max()))
    altitude_range = st.sidebar.slider(
        "Altitud (msnm)",
        min_value=float(round(altitude_bounds[0])),
        max_value=float(round(altitude_bounds[1])),
        value=(float(round(altitude_bounds[0])), float(round(altitude_bounds[1]))),
        step=25.0,
        key=f"{_KEY}_altitude",
        help="Restringe el análisis a las provincias situadas en la franja indicada.",
    )

    # --- Condiciones climáticas -------------------------------------------
    _group_label("Condiciones climáticas")

    seasons = st.sidebar.multiselect(
        "Temporada",
        options=frame["temporada"].cat.categories.tolist(),
        default=[],
        key=f"{_KEY}_seasons",
        placeholder="Todas las temporadas",
        help="Lluviosa (nov–mar), Transición (abr, oct) y Seca (may–sep).",
    )

    threshold_key = st.sidebar.radio(
        "Umbral de helada",
        options=list(FROST_THRESHOLD_LABELS),
        format_func=lambda key: FROST_THRESHOLD_LABELS[key],
        key=f"{_KEY}_threshold",
        help=(
            "El umbral meteorológico (0 °C) define el fenómeno físico; el "
            "agronómico (3 °C) marca el inicio del daño fisiológico en cultivos "
            "altoandinos sensibles. Cambiarlo recalcula todos los indicadores."
        ),
    )

    temperature_bounds = (
        float(frame["temperatura_minima"].min()),
        float(frame["temperatura_minima"].max()),
    )
    temperature_range = st.sidebar.slider(
        "Rango de temperatura mínima (°C)",
        min_value=float(round(temperature_bounds[0], 1)),
        max_value=float(round(temperature_bounds[1], 1)),
        value=(float(round(temperature_bounds[0], 1)), float(round(temperature_bounds[1], 1))),
        step=0.5,
        key=f"{_KEY}_temperature",
        help="Aísla franjas térmicas concretas para estudiar su composición.",
    )

    only_frost = st.sidebar.toggle(
        "Analizar sólo días con helada",
        value=False,
        key=f"{_KEY}_only_frost",
        help=(
            "Restringe el subconjunto a los eventos de helada. Útil para "
            "caracterizar su severidad sin que los días sin evento diluyan los promedios."
        ),
    )

    # --- Análisis ----------------------------------------------------------
    _group_label("Variable y calidad")

    analysis_variable = st.sidebar.selectbox(
        "Variable de análisis",
        options=[column for column in NUMERIC_ANALYSIS_VARIABLES if column in frame.columns],
        format_func=variable_label,
        key=f"{_KEY}_variable",
        help=(
            "Determina la variable representada en los gráficos configurables de "
            "distribución, evolución y comparación."
        ),
    )

    exclude_suspicious = st.sidebar.toggle(
        "Excluir registros sospechosos",
        value=False,
        key=f"{_KEY}_exclude",
        help=(
            "Descarta las observaciones marcadas por el control de calidad de la "
            "precipitación (acumulados diarios superiores a 100 mm, atribuibles al reanálisis)."
        ),
    )

    state = FilterState(
        date_range=date_range,
        provinces=tuple(provinces),
        ecological_tiers=tuple(tiers),
        seasons=tuple(seasons),
        min_temperature_range=temperature_range,
        altitude_range=altitude_range,
        frost_threshold=threshold_key,
        only_frost_days=only_frost,
        exclude_suspicious=exclude_suspicious,
        analysis_variable=analysis_variable,
    ).with_defaults(
        {
            "date_range": (minimum_date, maximum_date),
            "provinces": (),
            "ecological_tiers": (),
            "seasons": (),
            "min_temperature_range": (
                float(round(temperature_bounds[0], 1)),
                float(round(temperature_bounds[1], 1)),
            ),
            "altitude_range": (float(round(altitude_bounds[0])), float(round(altitude_bounds[1]))),
            "frost_threshold": "meteorologica",
            "only_frost_days": False,
            "exclude_suspicious": False,
            "analysis_variable": defaults.analysis_variable,
        }
    )

    filtered = apply_filters(frame, state)
    _render_summary(frame, filtered, state)

    # El estado se publica para que cualquier página pueda leerlo sin volver a
    # dibujar los controles.
    st.session_state["pn_active_filters"] = state
    return state, filtered


def _resolve_preset(
    preset: str, minimum_date: date, maximum_date: date
) -> tuple[date, date] | None:
    """Traduce un atajo temporal en un rango de fechas concreto.

    Args:
        preset: Nombre del atajo seleccionado.
        minimum_date: Primer día disponible.
        maximum_date: Último día disponible.

    Returns:
        El rango correspondiente, o ``None`` si el usuario eligió el modo
        personalizado y debe mostrarse el selector de fechas.
    """
    if preset == "Todo el periodo":
        return minimum_date, maximum_date
    if preset == "Últimos 3 años":
        return max(minimum_date, date(maximum_date.year - 2, 1, 1)), maximum_date
    if preset == "Últimos 5 años":
        return max(minimum_date, date(maximum_date.year - 4, 1, 1)), maximum_date
    if preset == "Última campaña agrícola":
        # La campaña agrícola peruana corre de agosto a julio.
        start_year = maximum_date.year if maximum_date.month >= 8 else maximum_date.year - 1
        campaign_start = date(start_year, 8, 1)
        return max(minimum_date, campaign_start), maximum_date
    return None


def _render_summary(original: pd.DataFrame, filtered: pd.DataFrame, state: FilterState) -> None:
    """Muestra el alcance de la selección activa al pie de la barra lateral.

    Informar del volumen retenido es lo que evita que el usuario interprete un
    subconjunto pequeño como si fuese el total.

    Args:
        original: Dataset completo.
        filtered: Subconjunto resultante.
        state: Selección activa.
    """
    impact = filter_impact(len(original), len(filtered))

    st.sidebar.markdown(
        f"""
        <div class="pn-filter-summary">
          <span class="pn-filter-summary__title">Selección activa</span>
          {format_integer(impact['retenidas'])} de {format_integer(len(original))} registros
          ({format_percent(impact['porcentaje_retenido'])})<br>
          <span style="opacity:.85">{state.describe()}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.sidebar.error(
            "La combinación de filtros no devuelve ningún registro. Amplíe alguno de los rangos."
        )
    elif impact["porcentaje_retenido"] < 2:
        st.sidebar.warning(
            "La selección retiene menos del 2 % de los datos. Las pruebas estadísticas "
            "pueden perder potencia con muestras tan reducidas."
        )

    if st.sidebar.button("Restablecer todos los filtros", key=f"{_KEY}_reset"):
        _reset_filters()


def _reset_filters() -> None:
    """Elimina el estado de los controles y recarga la aplicación."""
    for key in [key for key in st.session_state if key.startswith(_KEY)]:
        del st.session_state[key]
    st.rerun()


def get_active_filters() -> FilterState | None:
    """Recupera el estado de filtros publicado por la barra lateral.

    Returns:
        El estado activo, o ``None`` si la barra lateral aún no se ha dibujado.
    """
    state = st.session_state.get("pn_active_filters")
    return state if isinstance(state, FilterState) else None
