"""Vista general del dashboard de riesgo agroclimático por heladas en Puno."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import cards, charts, downloads, kpi, layout, sidebar
from utils.insights import (
    build_executive_summary,
    detect_insights,
    interpret_geographic,
    interpret_time_series,
)
from utils.io import load_clean_dataset
from utils.metrics import (
    aggregate_by_province,
    aggregate_monthly,
    compute_kpis,
)


layout.setup_page("Vista general")

try:
    frame = load_clean_dataset()
except (FileNotFoundError, ValueError) as error:
    layout.render_empty_state(
        "No se pudo cargar el conjunto de datos",
        f"Ejecute `python scripts/02_procesar_datos.py` y vuelva a abrir la aplicación. Detalle: {error}",
        icon="🗂️",
    )
    st.stop()

state, filtered = sidebar.render_sidebar(frame)
layout.render_header(compact=False)
layout.render_page_title(
    "Vista general",
    "Situación regional, indicadores dinámicos, estacionalidad y prioridades territoriales.",
    eyebrow="P1 · Resumen ejecutivo",
)

if filtered.empty:
    layout.render_empty_state(
        "La selección no contiene observaciones",
        "Amplíe el periodo o alguno de los rangos del panel lateral.",
    )
    st.stop()

indicators = compute_kpis(filtered, frost_column=state.frost_column)
kpi.render_kpi_row(indicators, columns=4)
kpi.render_kpi_help(indicators)

cards.render_section_header(
    "Resumen ejecutivo",
    "Síntesis calculada en tiempo real sobre los filtros activos.",
    eyebrow="P1",
    question="¿Cuál es la situación general del riesgo por heladas en la región Puno?",
)
with cards.card():
    st.markdown(build_executive_summary(filtered, frost_column=state.frost_column))

cards.render_section_header(
    "Hallazgos automáticos",
    "El motor analítico prioriza brechas, extremos, tendencias y alertas de calidad.",
    eyebrow="P1",
    question="¿Qué patrones merecen atención inmediata en la selección?",
)
cards.render_insights(
    detect_insights(filtered, frost_column=state.frost_column, limit=6), columns=2
)

province = aggregate_by_province(filtered, frost_column=state.frost_column)
monthly = aggregate_monthly(filtered, frost_column=state.frost_column)

cards.render_section_header(
    "Panel de riesgo",
    "Nivel medio regional y composición de la intensidad observada.",
    eyebrow="P1 · P4",
    question="¿Cuánto riesgo existe y cómo se distribuye su severidad?",
)
left, right = st.columns([1, 2], gap="large")
with left:
    risk_figure = charts.gauge_risk(
        float(filtered["indice_riesgo_helada"].mean()), title="Índice medio de riesgo (0–100)"
    )
    charts.render(risk_figure, height=360, key="home_gauge")
with right:
    intensity_figure = charts.bar_stacked_intensity(
        filtered, title="Composición de la intensidad por provincia", normalize=True
    )
    charts.render(intensity_figure, height=360, key="home_intensity")
cards.render_caption(
    1,
    "Nivel y composición del riesgo agroclimático",
    "El medidor resume el índice compuesto y las barras muestran proporciones comparables.",
)
intensity_table = pd.crosstab(
    filtered["provincia"], filtered["intensidad_helada"], normalize="index"
).mul(100).round(1).reset_index()
cards.render_table_view(intensity_table)

cards.render_section_header(
    "Prioridad territorial",
    "Ordenamiento por días que alcanzan el umbral de helada seleccionado.",
    eyebrow="P4",
    question="¿Qué provincias concentran la mayor y la menor exposición?",
)
ranking_figure = charts.bar_ranking(
    province,
    category="provincia",
    value="dias_helada",
    title="Días con helada por provincia",
    value_label="Días",
    orientation="h",
    show_values=True,
)
charts.render(ranking_figure, height=520, key="home_ranking")
cards.render_caption(
    2,
    "Ranking provincial de ocurrencia de heladas",
    "Los conteos responden al periodo, territorio y umbral seleccionados.",
)
cards.render_table_view(province)
cards.render_interpretation(interpret_geographic(province))

cards.render_section_header(
    "Ciclo anual",
    "La incidencia de heladas y la precipitación se muestran en paneles apilados para evitar un doble eje engañoso.",
    eyebrow="P2 · P3",
    question="¿En qué meses se concentra la ventana crítica regional?",
)
cycle_figure = charts.line_monthly_cycle(
    monthly,
    value="tasa_helada",
    secondary_value="precipitacion_media",
    title="Ciclo mensual de heladas y precipitación",
    y_label="Tasa de helada (%)",
)
charts.render(cycle_figure, height=520, key="home_cycle")
cards.render_caption(
    3,
    "Ciclo anual promedio",
    "La comparación revela la estacionalidad y su relación con el periodo seco.",
)
cards.render_table_view(monthly)
cards.render_interpretation(interpret_time_series(filtered, frost_column=state.frost_column))

cards.render_section_header(
    "Guía de navegación",
    "Cada página responde una parte distinta de la investigación.",
    eyebrow="Recorrido recomendado",
)
navigation = [
    ("1", "Análisis descriptivo", "Distribuciones, comparaciones, evolución y alertas."),
    ("2", "Análisis multidimensional", "Correlaciones, PCA, grupos, anomalías y modelo explicativo."),
    ("3", "Análisis geográfico", "Mapa interactivo, altitud, cuencas y efecto lacustre."),
    ("4", "Conclusiones", "Respuestas a P1–P6, recomendaciones y limitaciones."),
    ("5", "Metodología y datos", "Fuentes, limpieza, validación, diccionario y reproducibilidad."),
]
for column, item in zip(st.columns(5, gap="small"), navigation):
    with column:
        with cards.card():
            st.markdown(f"**{item[0]} · {item[1]}**")
            st.caption(item[2])

layout.render_research_questions()
layout.render_team()
downloads.render_download_panel(filtered, state, key_prefix="home")
layout.render_footer()
