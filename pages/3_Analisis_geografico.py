"""Distribución territorial y efecto moderador del lago Titicaca."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components import cards, charts, downloads, layout, sidebar
from config.settings import MONTH_ABBREV
from utils.insights import interpret_geographic
from utils.io import load_clean_dataset
from utils.metrics import aggregate_by_province
from utils.stats_tools import compare_groups, fit_linear_trend


layout.setup_page("Análisis geográfico")
frame = load_clean_dataset()
state, filtered = sidebar.render_sidebar(frame)
layout.render_header(compact=True)
layout.render_page_title(
    "Análisis geográfico",
    "Localización del riesgo, gradiente altitudinal, cuencas y efecto térmico del Titicaca.",
    eyebrow="P4 · Mapa interactivo",
)
if filtered.empty:
    layout.render_empty_state("Sin datos", "Amplíe los filtros del panel lateral.")
    st.stop()

province = aggregate_by_province(filtered, frost_column=state.frost_column)
zone_lookup = filtered.groupby("provincia", observed=True)["zona_agroecologica"].first()
province["zona_agroecologica"] = province["provincia"].map(zone_lookup)

cards.render_section_header(
    "Mapa provincial interactivo",
    "El color comunica magnitud y el tamaño refuerza una segunda métrica seleccionable.",
    eyebrow="P4",
    question="¿Dónde se concentra el riesgo agroclimático por heladas?",
)
color_options = {
    "dias_helada": "Días con helada",
    "tasa_helada": "Tasa de helada (%)",
    "riesgo_medio": "Índice medio de riesgo",
    "tmin_absoluta": "Temperatura mínima absoluta (°C)",
    "precipitacion_anual": "Precipitación anual (mm)",
    "altitud": "Altitud (msnm)",
}
left_control, right_control = st.columns(2)
with left_control:
    color_metric = st.selectbox(
        "Color del mapa", list(color_options), format_func=color_options.get, key="geo_color"
    )
with right_control:
    size_metric = st.selectbox(
        "Tamaño del punto",
        ["dias_helada", "riesgo_medio", "precipitacion_anual", "altitud"],
        format_func=color_options.get,
        key="geo_size",
    )
map_figure = charts.map_provinces(
    province,
    color=color_metric,
    size=size_metric,
    title="Capitales de las 13 provincias de Puno",
    color_label=color_options[color_metric],
)
charts.render(map_figure, height=640, key="geo_map")
cards.render_caption(1, "Mapa de riesgo provincial", "Cada punto representa la capital provincial en la celda ERA5-Land correspondiente.")
cards.render_table_view(province)
cards.render_interpretation(interpret_geographic(province))

cards.render_section_header(
    "Mapa de extremos térmicos",
    "La mínima absoluta aporta una vista complementaria: un evento extremo puede no coincidir con la mayor frecuencia.",
    eyebrow="P4",
    question="¿Las provincias con más heladas son también las que alcanzan los extremos más fríos?",
)
extreme_map = charts.map_provinces(
    province,
    color="tmin_absoluta",
    size="dias_helada",
    title="Temperatura mínima absoluta y frecuencia acumulada",
    color_label="Mínima absoluta (°C)",
)
charts.render(extreme_map, height=560, key="geo_extreme_map")
cards.render_caption(2, "Mapa complementario de extremos", "Color y tamaño separan intensidad extrema de frecuencia.")

cards.render_section_header(
    "Perfil altitudinal",
    "Las trece provincias se etiquetan directamente para identificar desviaciones respecto de la relación media.",
    eyebrow="P5",
    question="¿Cuánto explica la altitud y qué territorios contradicen ese patrón?",
)
altitude_scatter = charts.scatter_relationship(
    province,
    x="altitud",
    y="tasa_helada",
    title="Altitud frente a tasa de helada",
    size="dias_helada",
    show_regression=True,
    hover_extra=["provincia", "capital", "zona_agroecologica"],
)
if altitude_scatter.data:
    altitude_scatter.data[0].update(
        text=province["provincia"], mode="markers+text", textposition="top center"
    )
charts.render(altitude_scatter, height=580, key="geo_altitude")
cards.render_caption(3, "Gradiente altitudinal", "Las etiquetas revelan excepciones como Yunguyo y San Román a cotas casi idénticas.")
cards.render_table_view(province[["provincia", "altitud", "tasa_helada", "dias_helada", "zona_agroecologica"]])
gradient = fit_linear_trend(province["altitud"], province["tasa_helada"])
if gradient:
    cards.render_metric_grid(
        {
            "Pendiente por 100 m": f"{gradient.slope * 100:+.2f} puntos porcentuales",
            "R²": f"{gradient.r_squared:.3f}",
            "p-valor": f"{gradient.p_value:.3g}",
            "Provincias": str(gradient.n),
        }
    )

cards.render_section_header(
    "Cuencas, pisos y efecto lacustre",
    "La comparación no paramétrica contrasta la estructura territorial con el mecanismo físico esperado.",
    eyebrow="P4 · P5",
    question="¿La cuenca, el piso ecológico y la cercanía al lago modifican el régimen térmico?",
)
col1, col2 = st.columns(2, gap="large")
with col1:
    basin_box = charts.box_by_category(
        filtered,
        variable="temperatura_minima",
        category="cuenca",
        title="Temperatura mínima por cuenca",
    )
    charts.render(basin_box, height=430, key="geo_basin")
with col2:
    tier_violin = charts.violin_by_category(
        filtered,
        variable="temperatura_minima",
        category="piso_ecologico",
        title="Perfil térmico por piso ecológico",
    )
    charts.render(tier_violin, height=430, key="geo_tier")
cards.render_caption(4, "Contraste por unidades territoriales", "La forma completa de la distribución evita resumir territorios heterogéneos con una sola media.")

lake_data = filtered.assign(
    condicion_lacustre=filtered["zona_agroecologica"].astype(str).eq("Altiplano circunlacustre")
    .map({True: "Circunlacustre", False: "Resto de la región"})
)
lake_summary = (
    lake_data.groupby("condicion_lacustre", observed=True)
    .agg(
        registros=("fecha", "count"),
        tmin_media=("temperatura_minima", "mean"),
        tmin_absoluta=("temperatura_minima", "min"),
        tasa_helada=(state.frost_column, "mean"),
        oscilacion_media=("oscilacion_termica", "mean"),
    )
    .reset_index()
)
lake_summary["tasa_helada"] *= 100
lake_box = charts.box_by_category(
    lake_data,
    variable="temperatura_minima",
    category="condicion_lacustre",
    title="Efecto moderador del lago Titicaca",
)
charts.render(lake_box, height=430, key="geo_lake")
cards.render_caption(5, "Régimen circunlacustre frente al resto", "La gran masa de agua libera calor durante la noche y atenúa el enfriamiento radiativo.")
cards.render_table_view(lake_summary)
lake_test = compare_groups(lake_data, "temperatura_minima", "condicion_lacustre")
if lake_test:
    cards.render_metric_grid(
        {
            "Prueba": lake_test.test_name,
            "Estadístico": f"{lake_test.statistic:.2f}",
            "p-valor": f"{lake_test.p_value:.3g}",
            "Efecto ε²": f"{lake_test.effect_size:.3f}",
        }
    )
st.success(
    "Yunguyo (3 825 msnm) registra un régimen mucho más benigno que San Román (3 828 msnm). La diferencia de sólo tres metros descarta una explicación puramente altitudinal y es coherente con la inercia térmica del Titicaca."
)

cards.render_section_header(
    "Ventana crítica por territorio",
    "El índice compuesto se resume por provincia y mes con límites fijos comparables.",
    eyebrow="P4",
    question="¿En qué mes debe activarse la preparación en cada provincia?",
)
risk_calendar = filtered.pivot_table(
    index="provincia",
    columns="mes",
    values="indice_riesgo_helada",
    aggfunc="mean",
    observed=True,
).round(1)
risk_calendar = risk_calendar.reindex(sorted(risk_calendar.columns), axis=1)
risk_calendar.columns = [MONTH_ABBREV[int(month)] for month in risk_calendar.columns]
risk_calendar = risk_calendar.loc[risk_calendar.mean(axis=1).sort_values(ascending=False).index]
risk_heatmap = charts.heatmap_calendar(
    risk_calendar, title="Índice medio de riesgo provincia × mes", colorbar_title="Riesgo (0–100)"
)
charts.render(risk_heatmap, height=540, key="geo_risk_calendar")
cards.render_caption(6, "Calendario de riesgo agroclimático", "Los valores se pueden comparar entre selecciones porque el índice no usa percentiles relativos.")
cards.render_table_view(risk_calendar.reset_index())
cards.render_interpretation(interpret_geographic(province))

downloads.render_download_panel(
    filtered,
    state,
    extra_sheets={"resumen_territorial": province, "efecto_lacustre": lake_summary},
    key_prefix="geografico",
)
layout.render_footer()
