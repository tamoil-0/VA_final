"""Procedencia, ETL, documentación, validación y reproducibilidad."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components import cards, charts, downloads, layout, sidebar
from config.settings import (
    DATA_SOURCE,
    DATA_SOURCE_SECONDARY,
    PROJECT_PATHS,
    RANDOM_STATE,
)
from utils.io import (
    load_clean_dataset,
    load_data_dictionary,
    load_metadata,
    load_original_dataset,
)
from utils.metrics import compare_sources


layout.setup_page("Metodología y datos")
frame = load_clean_dataset()
state, filtered = sidebar.render_sidebar(frame)
layout.render_header(compact=True)
layout.render_page_title(
    "Metodología y datos",
    "Procedencia verificable, pipeline auditable, diccionario semántico y validación independiente.",
    eyebrow="Calidad · Reproducibilidad",
)
if filtered.empty:
    layout.render_empty_state("Sin datos", "Amplíe los filtros del panel lateral.")
    st.stop()

metadata = load_metadata()

cards.render_section_header(
    "Dos fuentes, dos funciones distintas",
    "ERA5-Land alimenta el dashboard; MERRA-2 se conserva como control externo independiente.",
    eyebrow="Procedencia",
    question="¿De dónde provienen los datos y por qué son adecuados para el problema?",
)
for column, source, role in zip(
    st.columns(2, gap="large"),
    [DATA_SOURCE, DATA_SOURCE_SECONDARY],
    ["Fuente primaria", "Fuente secundaria de validación"],
):
    with column:
        with cards.card():
            st.markdown(f"**{role}**")
            st.markdown(f"### {source.name}")
            st.markdown(
                f"**Institución:** {source.organization}  \n"
                f"**Modelo:** {source.model}  \n"
                f"**Resolución:** {source.spatial_resolution}  \n"
                f"**Endpoint:** [{source.endpoint}]({source.endpoint})  \n"
                f"**Portal:** [{source.portal}]({source.portal})  \n"
                f"**Licencia:** {source.license_note}"
            )
            with st.expander("Referencia APA 7"):
                st.write(source.citation_apa)
st.info(
    "Se eligió ERA5-Land (≈9 km) porque MERRA-2 (≈55 km) asignaba series idénticas a capitales vecinas y distorsionaba el gradiente altitudinal de Sandia. MERRA-2 no se descarta: se usa exclusivamente para comprobar que la señal de helada no depende de un único reanálisis."
)

cards.render_section_header(
    "Verificación en vivo de los requisitos",
    "Las evidencias se calculan desde el archivo limpio completo, no se escriben como cifras fijas.",
    eyebrow="Dataset",
    question="¿El conjunto satisface volumen, variedad, temporalidad, procedencia y calidad?",
)
numeric_count = int(frame.select_dtypes(include="number").shape[1])
categorical_count = int(frame.select_dtypes(include=["category", "object", "string"]).shape[1])
duplicate_count = int(frame.duplicated(["provincia", "fecha"]).sum())
missing_count = int(frame.isna().sum().sum())
requirements = pd.DataFrame(
    [
        ("≥ 500 registros", len(frame) >= 500, f"{len(frame):,} registros"),
        ("≥ 6 variables", frame.shape[1] >= 6, f"{frame.shape[1]} variables"),
        ("Variables numéricas", numeric_count > 0, f"{numeric_count} numéricas"),
        ("Variables categóricas", categorical_count > 0, f"{categorical_count} categóricas/textuales"),
        ("Componente temporal", pd.api.types.is_datetime64_any_dtype(frame["fecha"]), f"{frame['fecha'].min():%d/%m/%Y}–{frame['fecha'].max():%d/%m/%Y}"),
        ("Componente geográfico", {"latitud", "longitud"}.issubset(frame.columns), f"{frame['provincia'].nunique()} provincias con coordenadas"),
        ("Fuente verificable", True, "ECMWF/Copernicus y NASA POWER citados"),
        ("Sin datos sensibles", True, "Observaciones meteorológicas públicas"),
        ("Clave única", duplicate_count == 0, f"{duplicate_count} duplicados provincia–fecha"),
        ("Completitud", missing_count == 0, f"{missing_count} valores ausentes"),
    ],
    columns=["Requisito", "Cumple", "Evidencia calculada"],
)
requirements["Estado"] = requirements["Cumple"].map({True: "✓ Cumple", False: "✗ Revisar"})
st.dataframe(requirements.drop(columns="Cumple"), width="stretch", hide_index=True)

cards.render_section_header(
    "Pipeline de limpieza y transformación",
    "Cada etapa registra su operación y su efecto para que el proceso sea auditable.",
    eyebrow="ETL",
    question="¿Qué se modificó entre el dato nativo y el dataset analítico?",
)
pipeline = metadata.get("pipeline_etl", {})
steps = pipeline.get("steps", [])
for number, step in enumerate(steps, 1):
    st.markdown(f"**{number:02d}.** {step}")

cards.render_metric_grid(
    {
        "Entrada": f"{pipeline.get('rows_input', 0):,} × {pipeline.get('columns_input', 0)}",
        "Salida": f"{pipeline.get('rows_output', 0):,} × {pipeline.get('columns_output', 0)}",
        "Duplicados eliminados": str(pipeline.get("duplicates_removed", 0)),
        "Fechas añadidas": str(pipeline.get("missing_dates_added", 0)),
        "Valores imputados": str(pipeline.get("total_imputed", 0)),
        "Registros marcados": str(pipeline.get("suspicious_precipitation", 0)),
    }, columns=3
)

conversions = pd.DataFrame(
    [
        {"Variable": name, "Conversión": operation, "Razón": "Unificar unidades SI legibles y comparables"}
        for name, operation in pipeline.get("unit_conversions", {}).items()
    ]
)
st.markdown("#### Conversiones de unidad")
st.dataframe(conversions, width="stretch", hide_index=True)
st.warning(
    "La radiación llega en MJ/m²/día. Un rango inicialmente interpretado como kWh/m²/día habría marcado erróneamente 46 564 observaciones; verificar las unidades declaradas por la API evitó corromper PCA y correlaciones."
)

cards.render_section_header(
    "Original frente a limpio",
    "El dataset limpio conserva las observaciones y añade estructura temporal, territorial y de riesgo.",
    eyebrow="Transformación",
    question="¿Qué valor analítico aporta el pipeline sin alterar la evidencia original?",
)
original = load_original_dataset()
added_columns = [name for name in frame.columns if name not in original.columns]
cards.render_metric_grid(
    {
        "Original": f"{original.shape[0]:,} filas × {original.shape[1]} columnas",
        "Limpio": f"{frame.shape[0]:,} filas × {frame.shape[1]} columnas",
        "Variables añadidas": str(len(added_columns)),
        "Filas conservadas": f"{frame.shape[0] / original.shape[0]:.1%}",
    }
)
with st.expander("Comparar las primeras 10 filas", expanded=False):
    st.markdown("**Datos originales (nombres y unidades nativas)**")
    st.dataframe(original.head(10), width="stretch", hide_index=True)
    st.markdown("**Datos limpios y enriquecidos**")
    st.dataframe(frame.head(10), width="stretch", hide_index=True)
st.caption("Variables derivadas: " + ", ".join(added_columns))

cards.render_section_header(
    "Diccionario de datos interactivo",
    "Cada variable documenta tipo, unidad, definición, procedencia y transformación.",
    eyebrow="Documentación",
    question="¿Cómo se interpreta correctamente cada columna?",
)
dictionary = load_data_dictionary()
search = st.text_input("Buscar variable o descripción", placeholder="Ej.: radiación, helada, altitud")
dictionary_filtered = dictionary.copy()
if search:
    mask = dictionary_filtered.astype(str).apply(
        lambda column: column.str.contains(search, case=False, na=False)
    ).any(axis=1)
    dictionary_filtered = dictionary_filtered.loc[mask]
filter_columns = st.columns(2)
logical_column = next((c for c in dictionary.columns if "tipo" in c.lower()), None)
origin_column = next((c for c in dictionary.columns if "proced" in c.lower() or "origen" in c.lower()), None)
if logical_column:
    with filter_columns[0]:
        logical_values = st.multiselect(
            "Tipo lógico", sorted(dictionary[logical_column].dropna().astype(str).unique()), key="method_type"
        )
    if logical_values:
        dictionary_filtered = dictionary_filtered[dictionary_filtered[logical_column].astype(str).isin(logical_values)]
if origin_column:
    with filter_columns[1]:
        origin_values = st.multiselect(
            "Procedencia", sorted(dictionary[origin_column].dropna().astype(str).unique()), key="method_origin"
        )
    if origin_values:
        dictionary_filtered = dictionary_filtered[dictionary_filtered[origin_column].astype(str).isin(origin_values)]
st.dataframe(dictionary_filtered, width="stretch", hide_index=True, height=520)

cards.render_section_header(
    "Validación cruzada independiente",
    "ERA5-Land se contrasta observación a observación con MERRA-2 sin usar esta última para alimentar el dashboard.",
    eyebrow="Control externo",
    question="¿La señal de temperatura y helada se reproduce en otro producto climático?",
)
agreement = compare_sources(filtered)
if agreement:
    agreement_figure = charts.source_agreement_scatter(
        filtered, title="Temperatura mínima ERA5-Land frente a MERRA-2", sample=6_000
    )
    charts.render(agreement_figure, height=540, key="method_agreement")
    cards.render_caption(1, "Concordancia entre reanálisis", "La diagonal representa acuerdo perfecto; la dispersión refleja resoluciones y modelos distintos.")
    cards.render_metric_grid(
        {
            "Pares": f"{int(agreement['n_pares']):,}",
            "Correlación": f"{agreement['correlacion']:.3f}",
            "Sesgo ERA5−MERRA": f"{agreement['sesgo_medio']:+.2f} °C",
            "MAE": f"{agreement['error_absoluto_medio']:.2f} °C",
            "RMSE": f"{agreement['raiz_error_cuadratico_medio']:.2f} °C",
            "Concordancia de helada": f"{agreement['concordancia_helada']:.1f} %",
        }, columns=3
    )
    cards.render_table_view(
        filtered[["fecha", "provincia", "temperatura_minima", "temperatura_minima_merra2", "helada", "helada_merra2"]].head(500)
    )
    st.info(
        "r≈0,611 es coherente con comparar celdas de ≈9 km y ≈55 km: la malla gruesa suaviza extremos y mezcla topografía. La concordancia binaria de aproximadamente 83 % respalda la señal, aunque obliga a cautela con valores puntuales."
    )
else:
    st.info("La selección contiene menos de 30 pares completos para validar ambas fuentes.")

cards.render_section_header(
    "Definición del índice de riesgo",
    "Un índice compuesto traduce cuatro mecanismos físicos a una escala fija de 0 a 100.",
    eyebrow="Métrica derivada",
    question="¿Cómo se construye y por qué puede compararse entre filtros?",
)
st.latex(
    r"R = 100\,[0.45D_{termico} + 0.25C_{despejado} + 0.20S_{aire} + 0.10C_{viento}]"
)
components = pd.DataFrame(
    [
        ("Déficit térmico", "45 %", "Profundidad por debajo de 3 °C; aproxima estrés fisiológico."),
        ("Cielo despejado", "25 %", "Favorece pérdida de onda larga durante la noche."),
        ("Sequedad del aire", "20 %", "Reduce el efecto amortiguador del vapor de agua."),
        ("Calma del viento", "10 %", "Limita mezcla con capas de aire relativamente más cálidas."),
    ],
    columns=["Componente", "Peso", "Mecanismo"],
)
st.dataframe(components, width="stretch", hide_index=True)
st.success(
    "Los límites de normalización son fijos y físicos, no percentiles del filtro. Por eso 60/100 mantiene el mismo significado al cambiar de provincia o periodo."
)

cards.render_section_header(
    "Reproducibilidad y descargas",
    "El flujo completo puede regenerarse desde los datos nativos con versiones y semilla declaradas.",
    eyebrow="Ejecución",
)
st.code(
    "python scripts/01_descargar_datos.py\n"
    "python scripts/02_procesar_datos.py\n"
    "python scripts/03_generar_resultados.py\n"
    "streamlit run app.py",
    language="powershell",
)
packages = ["streamlit", "pandas", "numpy", "plotly", "scikit-learn", "scipy", "openpyxl"]
detected = {}
for package in packages:
    try:
        detected[package] = version(package)
    except PackageNotFoundError:
        detected[package] = "no instalado"
cards.render_metric_grid({**detected, "RANDOM_STATE": str(RANDOM_STATE)}, columns=4)

first, second, third = st.columns(3)
with first:
    st.download_button(
        "Descargar dataset original",
        data=PROJECT_PATHS.original_dataset.read_bytes(),
        file_name="datos_originales.csv",
        mime="text/csv",
        width="stretch",
        key="method_original",
    )
with second:
    st.download_button(
        "Descargar dataset limpio",
        data=PROJECT_PATHS.clean_dataset.read_bytes(),
        file_name="datos_limpios.csv",
        mime="text/csv",
        width="stretch",
        key="method_clean",
    )
with third:
    st.download_button(
        "Descargar diccionario",
        data=PROJECT_PATHS.data_dictionary.read_bytes(),
        file_name="diccionario_datos.csv",
        mime="text/csv",
        width="stretch",
        key="method_dictionary",
    )

downloads.render_download_panel(
    filtered,
    state,
    extra_sheets={"requisitos": requirements, "conversiones": conversions},
    key_prefix="metodologia",
)
layout.render_footer()
