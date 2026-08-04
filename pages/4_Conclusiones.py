"""Respuestas a las preguntas, conclusiones, recomendaciones y limitaciones."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components import cards, downloads, layout, sidebar
from config.settings import (
    DATA_SOURCE,
    DATA_SOURCE_SECONDARY,
    MULTIVARIATE_FEATURES,
    RANDOM_STATE,
    RESEARCH_QUESTIONS,
)
from utils.insights import build_executive_summary, detect_insights
from utils.io import load_clean_dataset
from utils.metrics import aggregate_annual, aggregate_by_province, aggregate_monthly
from utils.ml_models import detect_anomalies, run_kmeans, run_pca
from utils.stats_tools import mann_kendall_test, top_correlations


@st.cache_data(show_spinner=False, max_entries=8)
def _multivariate_summary(data: pd.DataFrame) -> dict[str, object]:
    sample = data.sample(12_000, random_state=RANDOM_STATE) if len(data) > 12_000 else data
    features = [name for name in MULTIVARIATE_FEATURES if name in sample.columns]
    pca = run_pca(sample, features, n_components=5)
    cluster = run_kmeans(sample, features, max_clusters=4)
    anomaly = detect_anomalies(sample, features, contamination=0.01)
    return {
        "n": len(sample),
        "pca_2": float(pca.explained_variance[:2].sum() * 100) if pca else None,
        "pca_90": pca.n_components_90 if pca else None,
        "k": cluster.n_clusters if cluster else None,
        "silueta": cluster.silhouette if cluster else None,
        "anomalias": anomaly.n_anomalies if anomaly else None,
    }


layout.setup_page("Conclusiones")
frame = load_clean_dataset()
state, filtered = sidebar.render_sidebar(frame)
layout.render_header(compact=True)
layout.render_page_title(
    "Conclusiones y decisiones",
    "Respuestas explícitas a P1–P6, recomendaciones por destinatario y límites de inferencia.",
    eyebrow="Síntesis de la investigación",
)
if filtered.empty:
    layout.render_empty_state("Sin datos", "Amplíe los filtros del panel lateral.")
    st.stop()

cards.render_section_header(
    "Resumen ejecutivo",
    "La redacción se recalcula sobre el subconjunto activo y conserva el umbral elegido.",
    eyebrow="P1–P6",
    question="¿Qué debe recordar un tomador de decisiones de este análisis?",
)
with cards.card():
    st.markdown(build_executive_summary(filtered, frost_column=state.frost_column))

province = aggregate_by_province(filtered, frost_column=state.frost_column)
monthly = aggregate_monthly(filtered, frost_column=state.frost_column)
annual = aggregate_annual(filtered, frost_column=state.frost_column)
mk = mann_kendall_test(annual["heladas_por_provincia"]) if len(annual) >= 4 else None
corr = top_correlations(
    filtered,
    ["temperatura_minima", "altitud", "humedad_relativa", "punto_rocio", "precipitacion", "radiacion_solar", "nubosidad"],
    target="temperatura_minima",
    limit=1,
)
multi = _multivariate_summary(filtered)

rate = filtered[state.frost_column].mean() * 100
top = province.iloc[0]
bottom = province.iloc[-1]
peak = monthly.loc[monthly["tasa_helada"].idxmax()]
severity = filtered["intensidad_helada"].value_counts()
severe = int(severity.get("Severa", 0) + severity.get("Extrema", 0))

answers = {
    "P1": (
        f"Se analizaron **{len(filtered):,} registros**, {filtered['fecha'].nunique():,} días y "
        f"{filtered['provincia'].nunique()} provincias. El **{rate:.1f} %** cruzó el umbral "
        f"de {state.frost_threshold_value:.0f} °C; la mínima media fue "
        f"**{filtered['temperatura_minima'].mean():.1f} °C** y el índice medio de riesgo "
        f"**{filtered['indice_riesgo_helada'].mean():.1f}/100**."
    ),
    "P2": (
        f"La ocurrencia se concentra en **{peak['mes_abrev']}**, con una tasa de "
        f"**{peak['tasa_helada']:.1f} %**. Se identificaron **{severe:,} eventos severos o "
        f"extremos**. La diferencia territorial y la estacionalidad muestran que una regla "
        "regional uniforme ocultaría ventanas de riesgo distintas."
    ),
    "P3": (
        (
            f"La prueba de Mann–Kendall estima una tendencia **{mk.trend}** con pendiente de "
            f"Sen **{mk.sen_slope:+.2f} días/año** (p={mk.p_value:.3g}). "
            + ("La evidencia es estadísticamente significativa." if mk.is_significant else "No se rechaza la ausencia de tendencia; domina la variabilidad interanual.")
        )
        if mk else "El filtro actual no conserva suficientes años para evaluar una tendencia robusta."
    ),
    "P4": (
        f"**{top['provincia']}** ocupa el primer lugar con **{int(top['dias_helada']):,} días** "
        f"({top['tasa_helada']:.1f} %), frente a **{bottom['provincia']}**, con "
        f"**{int(bottom['dias_helada']):,}** ({bottom['tasa_helada']:.1f} %). La brecha de "
        f"**{top['tasa_helada'] - bottom['tasa_helada']:.1f} puntos porcentuales** exige priorización territorial."
    ),
    "P5": (
        (
            f"La asociación de mayor magnitud con la temperatura mínima es "
            f"**{(corr[0].second if corr[0].first == 'temperatura_minima' else corr[0].first).replace('_', ' ')}**, "
            f"con r=**{corr[0].coefficient:+.3f}** (p={corr[0].p_value:.3g}). "
            "Las asociaciones describen mecanismos compatibles con enfriamiento radiativo, pero no prueban causalidad."
        )
        if corr else "El filtro actual no ofrece variabilidad suficiente para estimar asociaciones."
    ),
    "P6": (
        f"Sobre una muestra reproducible de **{multi['n']:,} observaciones**, las dos primeras "
        f"componentes explican **{multi['pca_2']:.1f} %** de la varianza y se necesitan "
        f"**{multi['pca_90']} componentes** para alcanzar 90 %. K-Means identifica "
        f"**{multi['k']} grupos** (silueta={multi['silueta']:.3f}) y Isolation Forest marca "
        f"**{multi['anomalias']} días atípicos**."
        if multi["pca_2"] is not None and multi["k"] is not None
        else "La selección actual es demasiado pequeña para ejecutar con estabilidad PCA y K-Means."
    ),
}

cards.render_section_header(
    "Respuesta a las seis preguntas de investigación",
    "Cada respuesta combina magnitudes calculadas y el método que permite sostenerla.",
    eyebrow="Resultados",
)
for question in RESEARCH_QUESTIONS:
    with st.expander(f"{question.code} · {question.question}", expanded=True):
        st.markdown(answers[question.code])
        st.caption(f"Método: {question.method} · Página: {question.page}")

cards.render_section_header(
    "Hallazgos priorizados",
    "Los hallazgos se ordenan por severidad y explican evidencia, decisión y cautela.",
    eyebrow="Interpretación",
)
cards.render_insights(
    detect_insights(filtered, frost_column=state.frost_column, limit=10), columns=1
)

cards.render_section_header(
    "Recomendaciones accionables",
    "Las acciones se asignan a quienes tienen capacidad concreta de ejecutarlas.",
    eyebrow="Decisión",
)
recommendations = {
    "Gobierno Regional de Puno": [
        f"Priorizar infraestructura de protección en {top['provincia']} y el siguiente cuartil del ranking.",
        "Financiar estaciones en fondos de valle que la malla de 9 km no resuelve.",
        "Usar un fondo de contingencia sensible a la variabilidad interanual y no sólo al promedio.",
    ],
    "Dirección Regional Agraria": [
        f"Activar alertas antes de {peak['mes_abrev']}, el mes de mayor incidencia en la selección.",
        "Adoptar el umbral agronómico de 3 °C para cultivos sensibles, además del meteorológico.",
        "Cruzar alertas térmicas con calendario de cultivos y etapa fenológica.",
    ],
    "Productores y organizaciones": [
        "Programar riego, coberturas y manejo de invernaderos según la ventana provincial, no regional.",
        "Registrar daño y rendimiento en campo para calibrar el índice compuesto.",
        "Usar alertas con texto e icono; el color por sí solo no debe comunicar urgencia.",
    ],
    "Academia y servicios climáticos": [
        "Ampliar la serie a 30 años antes de formular afirmaciones de cambio climático.",
        "Validar ERA5-Land con estaciones SENAMHI de superficie y cuantificar incertidumbre local.",
        "Evaluar modelos espacio-temporales y calibrar pesos del riesgo contra pérdida agrícola observada.",
    ],
}
columns = st.columns(2, gap="large")
for index, (audience, items) in enumerate(recommendations.items()):
    with columns[index % 2]:
        with cards.card():
            st.markdown(f"**{audience}**")
            for item in items:
                st.markdown(f"- {item}")

cards.render_section_header(
    "Limitaciones y trabajo futuro",
    "Delimitar la inferencia es parte del resultado, no una nota secundaria.",
    eyebrow="Alcance",
)
limitations = [
    "La malla de aproximadamente 9 km suaviza extremos de fondos de valle y laderas.",
    "ERA5-Land y MERRA-2 son reanálisis; no sustituyen observaciones de estación en superficie.",
    "Diez años describen variabilidad, pero no bastan para atribuir cambio climático; una normal climática requiere 30 años.",
    "Cada provincia está representada por un punto en su capital y no por toda su heterogeneidad topográfica.",
    "Un registro de precipitación superior a 100 mm se conserva marcado y puede excluirse con un filtro.",
    "La correlación entre fuentes es moderada debido, entre otros factores, a la resolución espacial distinta.",
    "No se dispone de rendimiento o pérdida agrícola, por lo que no se cuantifica impacto económico.",
    "El índice de riesgo usa pesos expertos fijos y aún no está calibrado contra daño de campo.",
]
for number, text in enumerate(limitations, 1):
    st.markdown(f"{number}. {text}")
st.markdown(
    "**Trabajo futuro.** Integrar estaciones SENAMHI, ampliar a 30 años, incorporar rendimiento de cultivos, modelar pronóstico con variables futuras y desplegar alertas georreferenciadas con validación de usuarios rurales."
)

cards.render_section_header(
    "Fuentes y trazabilidad",
    "La procedencia, los filtros y la semilla permiten reproducir cada cifra mostrada.",
    eyebrow="Reproducibilidad",
)
st.markdown(f"- {DATA_SOURCE.citation_apa}\n- {DATA_SOURCE_SECONDARY.citation_apa}")
cards.render_metric_grid(
    {
        "Selección": state.describe(),
        "Registros": f"{len(filtered):,}",
        "Versión del dataset": "ERA5-PUNO-2015-2024 v1.0",
        "Semilla": str(RANDOM_STATE),
        "Generado": datetime.now().astimezone().strftime("%d/%m/%Y %H:%M %Z"),
    },
    columns=2,
)

layout.render_team()
downloads.render_download_panel(
    filtered,
    state,
    extra_sheets={"respuestas_P1_P6": pd.DataFrame(
        [{"pregunta": key, "respuesta": value.replace("**", "")} for key, value in answers.items()]
    )},
    key_prefix="conclusiones",
)
layout.render_footer()
