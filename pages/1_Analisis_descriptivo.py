"""Distribuciones, comparaciones, evolución temporal y rankings."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components import cards, charts, downloads, layout, sidebar
from utils.insights import interpret_distribution, interpret_geographic, interpret_time_series
from utils.io import load_clean_dataset
from utils.metrics import (
    aggregate_annual,
    aggregate_by_province,
    aggregate_time_series,
    frost_calendar,
    summary_table,
)
from utils.stats_tools import compare_groups, fit_linear_trend, mann_kendall_test


layout.setup_page("Análisis descriptivo")
frame = load_clean_dataset()
state, filtered = sidebar.render_sidebar(frame)
layout.render_header(compact=True)
layout.render_page_title(
    "Análisis descriptivo",
    "Distribución, comparación territorial, evolución temporal y alertas del fenómeno.",
    eyebrow="P2 · P3 · P4",
)
if filtered.empty:
    layout.render_empty_state("Sin datos", "Amplíe los filtros del panel lateral.")
    st.stop()

tab_distribution, tab_comparison, tab_time, tab_ranking = st.tabs(
    ["Distribuciones", "Comparaciones", "Evolución temporal", "Ranking y alertas"]
)

with tab_distribution:
    variable = state.analysis_variable
    cards.render_section_header(
        "Forma de la distribución",
        "El histograma, la densidad y los percentiles permiten distinguir el comportamiento típico de los extremos.",
        eyebrow="P2",
        question=f"¿Cómo se distribuye {variable.replace('_', ' ')}?",
    )
    thresholds = (0.0, 3.0) if variable == "temperatura_minima" else ()
    figure = charts.histogram_distribution(
        filtered,
        variable=variable,
        title=f"Distribución de {variable.replace('_', ' ')}",
        thresholds=thresholds,
        show_kde=True,
    )
    charts.render(figure, height=460, key="desc_hist")
    cards.render_caption(1, "Distribución de la variable seleccionada", "Las líneas térmicas marcan 0 y 3 °C cuando corresponde.")
    descriptive = summary_table(
        filtered,
        [variable, "temperatura_minima", "precipitacion", "indice_riesgo_helada"],
    )
    cards.render_table_view(descriptive)
    cards.render_interpretation(interpret_distribution(filtered, variable))

    left, right = st.columns(2, gap="large")
    with left:
        box = charts.box_by_category(
            filtered,
            variable=variable,
            category="temporada",
            title="Variación por temporada",
        )
        charts.render(box, height=430, key="desc_box_season")
    with right:
        ecdf = charts.ecdf_chart(
            filtered,
            variable=variable,
            category="piso_ecologico",
            title="Distribución acumulada por piso ecológico",
        )
        charts.render(ecdf, height=430, key="desc_ecdf")
    cards.render_caption(2, "Comparación de distribuciones", "La caja resume cuartiles y la ECDF permite comparar toda la distribución.")
    cards.render_table_view(
        filtered.groupby(["temporada", "piso_ecologico"], observed=True)[variable]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

with tab_comparison:
    cards.render_section_header(
        "Contrastes territoriales y estacionales",
        "Se combinan vistas de distribución, conteo y una prueba no paramétrica.",
        eyebrow="P2 · P4",
        question="¿Las condiciones térmicas difieren entre provincias, pisos y temporadas?",
    )
    province_order = (
        filtered.groupby("provincia", observed=True)["temperatura_minima"]
        .median().sort_values().index.astype(str).tolist()
    )
    box_province = charts.box_by_category(
        filtered,
        variable="temperatura_minima",
        category="provincia",
        order=province_order,
        title="Temperatura mínima por provincia",
    )
    charts.render(box_province, height=520, key="desc_box_province")
    cards.render_caption(3, "Distribución térmica provincial", "El orden por mediana evita una comparación alfabética sin sentido analítico.")
    cards.render_table_view(
        filtered.groupby("provincia", observed=True)["temperatura_minima"]
        .agg(n="count", media="mean", mediana="median", minima="min", maxima="max")
        .reset_index()
    )

    grouped = (
        filtered.groupby(["provincia", "temporada"], observed=True)[state.frost_column]
        .sum().rename("dias_helada").reset_index()
    )
    grouped_figure = charts.bar_grouped(
        grouped,
        category="provincia",
        value="dias_helada",
        series="temporada",
        title="Días con helada por provincia y temporada",
        value_label="Días",
    )
    charts.render(grouped_figure, height=500, key="desc_grouped")
    cards.render_caption(4, "Ocurrencia por provincia y temporada", "Las tres temporadas separan el efecto estacional del territorial.")
    cards.render_table_view(grouped)

    comparison_group = st.selectbox(
        "Variable de agrupación para el contraste",
        ["provincia", "temporada", "piso_ecologico"],
        key="desc_group_test",
    )
    result = compare_groups(filtered, "temperatura_minima", comparison_group)
    if result:
        cards.render_metric_grid(
            {
                "Prueba": result.test_name,
                "Grupos": str(result.n_groups),
                "Estadístico": f"{result.statistic:.2f}",
                "p-valor": f"{result.p_value:.3g}",
                "Tamaño del efecto ε²": f"{result.effect_size:.3f}",
            },
            columns=5,
        )
        st.info(
            f"{result.interpretation} Se aplica Kruskal–Wallis porque las series diarias no garantizan normalidad ni homocedasticidad; un ANOVA sería menos robusto."
        )

with tab_time:
    cards.render_section_header(
        "Evolución y tendencia",
        "La escala mensual revela estacionalidad; la anual permite contrastar una tendencia de fondo.",
        eyebrow="P3",
        question="¿La frecuencia de heladas cambia significativamente durante 2015–2024?",
    )
    monthly_series = aggregate_time_series(
        filtered, value_column=state.frost_column, frequency="MS", aggregation="mean"
    )
    monthly_series["valor"] = monthly_series["valor"] * 100
    timeline = charts.line_series(
        monthly_series,
        x="fecha",
        y="valor",
        title="Tasa mensual de helada",
        y_label="Tasa de helada (%)",
        fill=True,
    )
    charts.render(timeline, height=410, key="desc_timeline")
    cards.render_caption(5, "Serie mensual de incidencia", "La agregación reduce el ruido diario y conserva la ventana estacional.")
    cards.render_table_view(monthly_series)

    annual = aggregate_annual(filtered, frost_column=state.frost_column)
    annual_figure = charts.line_annual_with_trend(
        annual,
        value="heladas_por_provincia",
        title="Heladas anuales por provincia con OLS y pendiente de Sen",
        y_label="Días por provincia",
    )
    charts.render(annual_figure, height=430, key="desc_annual")
    cards.render_caption(6, "Tendencia interanual", "OLS estima una pendiente lineal y Mann–Kendall/Sen aportan un contraste robusto no paramétrico.")
    cards.render_table_view(annual)
    trend = fit_linear_trend(annual["anio"], annual["heladas_por_provincia"])
    mk = mann_kendall_test(annual["heladas_por_provincia"])
    if trend and mk:
        cards.render_metric_grid(
            {
                "Pendiente OLS": f"{trend.slope:+.2f} días/año",
                "R²": f"{trend.r_squared:.3f}",
                "p OLS": f"{trend.p_value:.3g}",
                "Pendiente de Sen": f"{mk.sen_slope:+.2f} días/año",
                "p Mann–Kendall": f"{mk.p_value:.3g}",
            }, columns=5
        )

    calendar = frost_calendar(filtered, frost_column=state.frost_column)
    heatmap = charts.heatmap_calendar(
        calendar, title="Calendario provincia × mes", colorbar_title="Helada (%)"
    )
    charts.render(heatmap, height=520, key="desc_calendar")
    cards.render_caption(7, "Calendario territorial de heladas", "Cada celda muestra la proporción de días que cruza el umbral.")
    cards.render_table_view(calendar.reset_index())
    cards.render_interpretation(interpret_time_series(filtered, frost_column=state.frost_column))

with tab_ranking:
    cards.render_section_header(
        "Ranking y alertas operativas",
        "La métrica puede cambiar sin perder el orden visual ni la trazabilidad.",
        eyebrow="P4",
        question="¿Dónde debe priorizarse la gestión del riesgo?",
    )
    province = aggregate_by_province(filtered, frost_column=state.frost_column)
    metric_labels = {
        "dias_helada": "Días con helada",
        "tasa_helada": "Tasa de helada (%)",
        "riesgo_medio": "Índice medio de riesgo",
        "deficit_termico_total": "Déficit térmico acumulado",
    }
    metric = st.selectbox(
        "Métrica del ranking", list(metric_labels), format_func=metric_labels.get, key="desc_rank_metric"
    )
    rank_figure = charts.bar_ranking(
        province.sort_values(metric, ascending=False),
        category="provincia",
        value=metric,
        title=metric_labels[metric],
        value_label=metric_labels[metric],
        orientation="h",
        color_by_value=True,
    )
    charts.render(rank_figure, height=520, key="desc_ranking")
    cards.render_caption(8, "Priorización provincial", "La escala secuencial codifica magnitud, no identidad nominal.")

    alert_rows = []
    for row in province.itertuples(index=False):
        if row.tasa_helada >= 30:
            level, severity, icon = "Crítica", "critical", "⚠️"
        elif row.tasa_helada >= 15:
            level, severity, icon = "Alta", "serious", "▲"
        elif row.tasa_helada >= 5:
            level, severity, icon = "Vigilancia", "warning", "●"
        else:
            level, severity, icon = "Baja", "good", "✓"
        alert_rows.append(
            {
                "Provincia": row.provincia,
                "Tasa (%)": row.tasa_helada,
                "Días": row.dias_helada,
                "Nivel": level,
                "Estado": cards.render_badge(level, severity, icon=icon),
            }
        )
    alert_frame = pd.DataFrame(alert_rows)
    st.markdown("#### Sistema de alertas")
    for row in alert_rows:
        st.markdown(
            f"{row['Estado']} **{row['Provincia']}** · {row['Tasa (%)']:.1f} % · {row['Días']:,} días",
            unsafe_allow_html=True,
        )
    cards.render_table_view(alert_frame.drop(columns="Estado"))
    cards.render_interpretation(interpret_geographic(province))

downloads.render_download_panel(filtered, state, key_prefix="descriptivo")
layout.render_footer()
