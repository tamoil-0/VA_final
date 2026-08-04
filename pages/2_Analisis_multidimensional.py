"""Técnicas multidimensionales y modelos explicativos del riesgo de helada."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from components import cards, charts, downloads, layout, sidebar
from config.settings import MULTIVARIATE_FEATURES, RANDOM_STATE
from utils.insights import interpret_multivariate
from utils.io import load_clean_dataset
from utils.metrics import aggregate_by_province
from utils.ml_models import (
    detect_anomalies,
    run_hierarchical_clustering,
    run_kmeans,
    run_pca,
    train_frost_classifier,
)
from utils.stats_tools import (
    correlation_matrix,
    correlation_pvalues,
    fit_linear_trend,
    modified_z_scores,
    top_correlations,
    variance_inflation_factors,
)


@st.cache_data(show_spinner=False, max_entries=8)
def _advanced_results(data: pd.DataFrame, frost_column: str):
    """Calcula los modelos costosos sobre una muestra reproducible y suficiente."""
    analysis = (
        data.sample(15_000, random_state=RANDOM_STATE).sort_index()
        if len(data) > 15_000
        else data.copy()
    )
    features = [name for name in MULTIVARIATE_FEATURES if name in analysis.columns]
    pca = run_pca(analysis, features, n_components=5)
    clustering = run_kmeans(analysis, features, max_clusters=4)
    anomalies = detect_anomalies(analysis, features, contamination=0.01)
    predictors = [
        "temperatura_media", "temperatura_maxima", "oscilacion_termica",
        "humedad_relativa", "punto_rocio", "nubosidad", "radiacion_solar",
        "velocidad_viento", "presion_superficie", "precipitacion", "altitud",
        "mes", "dia_anio",
    ]
    classifier = train_frost_classifier(analysis, predictors, target=frost_column)
    return analysis, pca, clustering, anomalies, classifier


layout.setup_page("Análisis multidimensional")
frame = load_clean_dataset()
state, filtered = sidebar.render_sidebar(frame)
layout.render_header(compact=True)
layout.render_page_title(
    "Análisis multidimensional",
    "Relaciones, factores latentes, regímenes homogéneos, anomalías y variables predictoras.",
    eyebrow="P5 · P6 · Técnicas avanzadas",
)
if filtered.empty:
    layout.render_empty_state("Sin datos", "Amplíe los filtros del panel lateral.")
    st.stop()

with st.spinner("Calculando modelos reproducibles…"):
    analysis, pca, clustering, anomalies, classifier = _advanced_results(
        filtered, state.frost_column
    )
if len(filtered) > len(analysis):
    st.caption(
        f"PCA, agrupamiento, anomalías y Random Forest usan una muestra reproducible de {len(analysis):,} de {len(filtered):,} registros (semilla {RANDOM_STATE}) para mantener la interfaz fluida."
    )

tab_corr, tab_pca, tab_cluster, tab_anomaly, tab_model = st.tabs(
    ["Correlaciones", "Componentes principales", "Agrupamiento", "Anomalías", "Modelo predictivo"]
)

correlation_features = [
    "temperatura_minima", "altitud", "humedad_relativa", "punto_rocio",
    "precipitacion", "radiacion_solar", "nubosidad", "velocidad_viento",
]

with tab_corr:
    cards.render_section_header(
        "Relaciones entre variables",
        "Los coeficientes se acompañan de significancia y magnitud para evitar confundir detectabilidad con relevancia.",
        eyebrow="P5",
        question="¿Qué relación existe entre altitud, humedad, radiación, precipitación y temperatura mínima?",
    )
    method = st.radio(
        "Método de correlación",
        ["pearson", "spearman"],
        horizontal=True,
        format_func=lambda item: "Pearson (lineal)" if item == "pearson" else "Spearman (monótona y robusta)",
        key="multi_corr_method",
    )
    corr = correlation_matrix(filtered, correlation_features, method=method)
    pvalues = correlation_pvalues(filtered, correlation_features, method=method)
    corr_figure = charts.heatmap_correlation(
        corr, pvalues, title=f"Matriz de correlación {method.capitalize()}"
    )
    charts.render(corr_figure, height=620, key="multi_corr")
    cards.render_caption(1, "Matriz de correlación", "Las celdas no significativas se marcan; el color divergente conserva el signo.")
    cards.render_table_view(corr.reset_index(names="variable"))
    st.warning(
        "Con decenas de miles de observaciones, una correlación muy pequeña puede resultar significativa. La interpretación debe priorizar |r| y el mecanismo físico, no sólo el p-valor."
    )

    x_variable = st.selectbox("Variable explicativa", correlation_features[1:], key="multi_x")
    y_variable = st.selectbox("Variable respuesta", correlation_features, key="multi_y")
    scatter = charts.scatter_relationship(
        filtered,
        x=x_variable,
        y=y_variable,
        color="temperatura_minima" if y_variable != "temperatura_minima" else None,
        title=f"{x_variable.replace('_', ' ')} frente a {y_variable.replace('_', ' ')}",
        show_regression=True,
        sample=6_000,
        hover_extra=["provincia", "fecha"],
    )
    charts.render(scatter, height=500, key="multi_scatter")
    cards.render_caption(2, "Relación configurable", "La recta resume la asociación lineal y no implica causalidad.")

    findings = top_correlations(
        filtered, correlation_features, target="temperatura_minima", limit=7, method=method
    )
    findings_frame = pd.DataFrame(
        {
            "Variable 1": [item.first for item in findings],
            "Variable 2": [item.second for item in findings],
            "Coeficiente": [item.coefficient for item in findings],
            "p-valor": [item.p_value for item in findings],
            "Magnitud": [item.strength for item in findings],
        }
    )
    vif = variance_inflation_factors(filtered, correlation_features)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Asociaciones principales")
        st.dataframe(findings_frame, width="stretch", hide_index=True)
    with right:
        st.markdown("#### Factor de inflación de varianza")
        st.dataframe(vif, width="stretch", hide_index=True)
    st.info("El VIF revela variables redundantes. Esta comprobación justifica reducir dimensiones antes de interpretar perfiles multivariantes.")

with tab_pca:
    cards.render_section_header(
        "Análisis de componentes principales",
        "PCA resume doce variables estandarizadas en factores ortogonales comparables.",
        eyebrow="P6",
        question="¿Cuántos factores latentes explican el sistema agroclimático y qué variables los definen?",
    )
    if pca is None:
        st.info("La selección requiere al menos 30 observaciones completas y variables con variación.")
    else:
        left, right = st.columns(2, gap="large")
        with left:
            scree = charts.pca_scree(pca, title="Varianza explicada y acumulada")
            charts.render(scree, height=430, key="multi_pca_scree")
        with right:
            loadings = charts.pca_loadings_heatmap(pca, title="Cargas de las componentes")
            charts.render(loadings, height=430, key="multi_pca_loadings")
        cards.render_caption(3, "Diagnóstico del PCA", "El scree determina cuántas componentes conservar y las cargas explican su significado.")
        cards.render_table_view(pca.loadings.reset_index(names="variable"))

        colors = analysis.loc[pca.row_index, "temperatura_minima"].to_numpy()
        biplot = charts.pca_biplot(
            pca,
            color_values=colors,
            color_label="Temperatura mínima (°C)",
            title="Proyección PCA y vectores de carga",
            sample=4_000,
        )
        charts.render(biplot, height=560, key="multi_pca_biplot")
        cards.render_caption(4, "Biplot de las dos primeras componentes", "La proximidad representa perfiles climáticos semejantes y los vectores indican dirección de asociación.")

        if pca.scores.shape[1] >= 3:
            plot_3d = charts.pca_scatter_3d(
                pca,
                color_values=colors,
                color_label="Temperatura mínima (°C)",
                title="Espacio tridimensional CP1–CP3",
                sample=4_000,
            )
            charts.render(plot_3d, height=620, key="multi_pca_3d")
            cards.render_caption(5, "Visualización tridimensional del PCA", "La tercera componente permite inspeccionar separaciones ocultas en el plano.")
        st.success(f"{pca.interpret_component(0)} {pca.interpret_component(1)}")
        st.info("Se estandariza porque la altitud se mide en miles de metros y dominaría artificialmente a temperaturas expresadas en decenas de grados.")
        cards.render_interpretation(interpret_multivariate(analysis, pca, clustering))

with tab_cluster:
    cards.render_section_header(
        "Regímenes agroclimáticos homogéneos",
        "K-Means identifica perfiles recurrentes; Ward muestra cómo se relacionan las trece provincias.",
        eyebrow="P6",
        question="¿Existen grupos climáticos diferenciados y estables en el periodo?",
    )
    if pca is None or clustering is None:
        st.info("Amplíe la selección para ejecutar el agrupamiento.")
    else:
        diagnostics = charts.elbow_silhouette(clustering.diagnostics, title="Selección del número de grupos")
        charts.render(diagnostics, height=500, key="multi_cluster_diag")
        cards.render_caption(6, "Codo y coeficiente de silueta", "Se elige el k que maximiza separación sin fragmentar en exceso.")
        cards.render_table_view(clustering.diagnostics)

        cluster_figure = charts.cluster_scatter(
            pca,
            clustering.labels,
            title=f"K-Means en el plano PCA (k={clustering.n_clusters})",
            cluster_names=clustering.profile.set_index("grupo")["etiqueta"].to_dict(),
            sample=5_000,
        )
        charts.render(cluster_figure, height=540, key="multi_cluster_scatter")
        cards.render_caption(7, "Separación de regímenes", "El tope de cuatro grupos conserva distinción cromática accesible entre cualquier par.")

        radar_features = [
            name for name in ["tmin_media", "oscilacion_media", "precipitacion_media", "humedad_media", "riesgo_medio"]
            if name in clustering.profile.columns
        ]
        if radar_features:
            radar = charts.cluster_radar(
                clustering.profile,
                features=radar_features,
                title="Perfil comparado de los grupos",
                cluster_names=clustering.profile.set_index("grupo")["etiqueta"].to_dict(),
            )
            charts.render(radar, height=530, key="multi_cluster_radar")
        st.dataframe(clustering.profile, width="stretch", hide_index=True)

        province = aggregate_by_province(filtered, frost_column=state.frost_column)
        hierarchy_features = ["tasa_helada", "tmin_media", "precipitacion_anual", "altitud", "riesgo_medio"]
        hierarchy = run_hierarchical_clustering(province, hierarchy_features, n_clusters=4)
        if hierarchy:
            dendrogram = charts.dendrogram_chart(hierarchy, title="Dendrograma provincial por enlace de Ward")
            charts.render(dendrogram, height=520, key="multi_dendrogram")
            membership = pd.DataFrame({"provincia": hierarchy.province_names, "grupo_ward": hierarchy.labels})
            cards.render_caption(8, "Jerarquía territorial", "Ward agrupa perfiles provinciales estandarizados minimizando la varianza interna.")
            cards.render_table_view(membership)
        cards.render_interpretation(interpret_multivariate(analysis, pca, clustering))

with tab_anomaly:
    cards.render_section_header(
        "Detección multivariante de anomalías",
        "Isolation Forest localiza combinaciones inusuales y la Z modificada aporta un contraste robusto univariante.",
        eyebrow="P6",
        question="¿Qué días presentan combinaciones climáticas atípicas?",
    )
    if anomalies is None:
        st.info("Se requieren al menos 100 observaciones completas.")
    else:
        timeline = charts.anomaly_timeline(
            analysis, anomalies.flags, variable="temperatura_minima", title="Anomalías sobre la serie de temperatura mínima"
        )
        charts.render(timeline, height=500, key="multi_anomaly")
        cards.render_caption(9, "Días atípicos multivariantes", "Los puntos se detectan por su combinación conjunta, no por un umbral aislado.")
        cards.render_metric_grid(
            {
                "Observaciones evaluadas": f"{len(analysis):,}",
                "Anomalías": f"{anomalies.n_anomalies:,}",
                "Contaminación": f"{anomalies.contamination * 100:.1f} %",
                "Variables": str(len(anomalies.features)),
            }
        )
        z_scores = modified_z_scores(analysis["temperatura_minima"])
        st.caption(f"La Z modificada (mediana y MAD) marca {(z_scores.abs() > 3.5).sum():,} extremos con |Z| > 3,5; es menos sensible a los propios atípicos que media y desviación.")
        cards.render_table_view(anomalies.extremes, label="Ver las 15 anomalías más extremas")

with tab_model:
    cards.render_section_header(
        "Random Forest explicativo",
        "Una partición temporal evalúa capacidad de discriminación y la importancia por permutación ordena predictores.",
        eyebrow="P6",
        question="¿Qué variables anticipan mejor la ocurrencia de una helada?",
    )
    if classifier is None:
        st.info("La selección debe contener al menos 500 registros y ambas clases del objetivo.")
    else:
        cards.render_metric_grid(
            {
                "AUC ROC": f"{classifier.roc_auc:.3f}",
                "Precisión media": f"{classifier.average_precision:.3f}",
                "Exactitud": f"{classifier.accuracy:.1%}",
                "Sensibilidad": f"{classifier.recall:.1%}",
                "Precisión": f"{classifier.precision:.1%}",
                "Entrenamiento": f"{classifier.n_train:,}",
                "Prueba": f"{classifier.n_test:,}",
                "Prevalencia": f"{classifier.positive_rate:.1%}",
            }
        )
        importance = charts.importance_bar(classifier.importances, title="Importancia por permutación", top_n=12)
        charts.render(importance, height=480, key="multi_model_importance")
        cards.render_caption(10, "Variables predictoras", "La caída del desempeño al permutar una variable cuantifica su aporte real.")
        cards.render_table_view(classifier.importances)
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            charts.render(charts.roc_curve_chart(classifier, title="Curva ROC"), height=350, key="multi_roc")
        with col2:
            charts.render(charts.precision_recall_chart(classifier, title="Curva precisión–recobrado"), height=350, key="multi_pr")
        with col3:
            charts.render(charts.confusion_heatmap(classifier, title="Matriz de confusión"), height=350, key="multi_confusion")
        st.info(
            "La partición es cronológica para reducir fuga por autocorrelación. Se excluyen temperatura mínima, déficit térmico e índice de riesgo porque definen el objetivo; incluirlos convertiría el problema en una tautología. El modelo es explicativo, no un pronóstico operativo."
        )

downloads.render_download_panel(
    filtered,
    state,
    extra_sheets={
        "pca_cargas": pca.loadings.reset_index(names="variable") if pca else pd.DataFrame(),
        "grupos": clustering.profile if clustering else pd.DataFrame(),
        "anomalias": anomalies.extremes if anomalies else pd.DataFrame(),
        "importancias": classifier.importances if classifier else pd.DataFrame(),
    },
    key_prefix="multidimensional",
)
layout.render_footer()
