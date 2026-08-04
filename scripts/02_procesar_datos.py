"""Paso 2 del flujo de datos: limpieza, transformación y documentación.

Uso::

    python scripts/02_procesar_datos.py

Produce:
    * ``data/datos_limpios.csv``     — dataset analítico final.
    * ``data/diccionario_datos.csv`` — diccionario de variables con estadísticos.
    * ``data/metadata.json``         — trazabilidad completa del proceso ETL.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (  # noqa: E402
    AUTHORS,
    COURSE,
    DATA_SOURCE,
    DATA_SOURCE_SECONDARY,
    ERA5_MODEL,
    FROST_THRESHOLDS,
    LOCAL_TIMEZONE,
    PROBLEM_STATEMENT,
    PROJECT_PATHS,
    PROVINCES,
    RESEARCH_QUESTIONS,
    STUDY_PERIOD,
)
from utils.preprocessing import build_data_dictionary, clean_dataset  # noqa: E402

#: Ruta del dataset secundario empleado en la validación cruzada.
VALIDATION_DATASET = PROJECT_PATHS.data / "datos_originales_validacion_merra2.csv"


def build_metadata(clean: pd.DataFrame, report_dict: dict) -> dict:
    """Compone el documento de metadatos del proyecto.

    Args:
        clean: Dataset limpio ya procesado.
        report_dict: Bitácora del pipeline serializada.

    Returns:
        Diccionario con la ficha técnica completa del conjunto de datos.
    """
    numeric = clean.select_dtypes(include=["number"]).columns
    categorical = clean.select_dtypes(include=["category", "object"]).columns
    boolean = clean.select_dtypes(include=["bool"]).columns

    return {
        "proyecto": {
            "titulo": COURSE["project_title"],
            "subtitulo": COURSE["project_subtitle"],
            "problema": PROBLEM_STATEMENT,
            "curso": f"{COURSE['course']} ({COURSE['course_code']})",
            "escuela": COURSE["school"],
            "facultad": COURSE["faculty"],
            "universidad": COURSE["university"],
            "docente": COURSE["professor"],
            "semestre": COURSE["semester"],
            "integrantes": [
                {"nombre": author.full_name, "rol": author.role, "aportes": list(author.contributions)}
                for author in AUTHORS
            ],
        },
        "fuente_primaria": {
            "nombre": DATA_SOURCE.name,
            "organizacion": DATA_SOURCE.organization,
            "endpoint": DATA_SOURCE.endpoint,
            "portal": DATA_SOURCE.portal,
            "modelo": DATA_SOURCE.model,
            "modelo_solicitado": ERA5_MODEL,
            "zona_horaria_agregacion": LOCAL_TIMEZONE,
            "resolucion_espacial": DATA_SOURCE.spatial_resolution,
            "licencia": DATA_SOURCE.license_note,
            "cita_apa": DATA_SOURCE.citation_apa,
        },
        "fuente_secundaria": {
            "nombre": DATA_SOURCE_SECONDARY.name,
            "organizacion": DATA_SOURCE_SECONDARY.organization,
            "modelo": DATA_SOURCE_SECONDARY.model,
            "resolucion_espacial": DATA_SOURCE_SECONDARY.spatial_resolution,
            "proposito": "Validación cruzada independiente del régimen de heladas.",
            "cita_apa": DATA_SOURCE_SECONDARY.citation_apa,
        },
        "cobertura": {
            "periodo_inicio": STUDY_PERIOD.start.isoformat(),
            "periodo_fin": STUDY_PERIOD.end.isoformat(),
            "anios": STUDY_PERIOD.n_years,
            "dias_por_provincia": int(clean.groupby("provincia", observed=True).size().max()),
            "provincias": [province.name for province in PROVINCES],
            "n_provincias": len(PROVINCES),
            "ambito": "Región Puno, Perú",
        },
        "estructura": {
            "n_registros": int(len(clean)),
            "n_variables": int(len(clean.columns)),
            "n_variables_numericas": int(len(numeric)),
            "n_variables_categoricas": int(len(categorical)),
            "n_variables_booleanas": int(len(boolean)),
            "clave_primaria": ["provincia", "fecha"],
            "granularidad": "una observación por provincia y día",
        },
        "umbrales": {
            "helada_meteorologica_c": FROST_THRESHOLDS["meteorologica"],
            "helada_agronomica_c": FROST_THRESHOLDS["agronomica"],
        },
        "preguntas_analisis": [
            {"codigo": q.code, "pregunta": q.question, "pagina": q.page, "metodo": q.method}
            for q in RESEARCH_QUESTIONS
        ],
        "pipeline_etl": report_dict,
        "generado_en": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }


def main() -> int:
    """Ejecuta el pipeline y persiste todos los artefactos de datos.

    Returns:
        ``0`` si el proceso concluyó correctamente.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    if not PROJECT_PATHS.original_dataset.exists():
        print(f"ERROR: no se encontró {PROJECT_PATHS.original_dataset}")
        print("Ejecute primero: python scripts/01_descargar_datos.py")
        return 1

    print("=" * 78)
    print("LIMPIEZA, TRANSFORMACIÓN Y DOCUMENTACIÓN DEL CONJUNTO DE DATOS")
    print("=" * 78)

    raw = pd.read_csv(PROJECT_PATHS.original_dataset)

    validation: pd.DataFrame | None = None
    if VALIDATION_DATASET.exists():
        validation = pd.read_csv(VALIDATION_DATASET)
        print(f"Fuente secundaria detectada: {len(validation):,} filas para validación cruzada.")
    else:
        print("AVISO: no se halló el dataset de validación; se omite la integración cruzada.")

    clean, report = clean_dataset(raw, validation)

    clean.to_csv(PROJECT_PATHS.clean_dataset, index=False, encoding="utf-8")
    dictionary = build_data_dictionary(clean)
    dictionary.to_csv(PROJECT_PATHS.data_dictionary, index=False, encoding="utf-8")

    metadata = build_metadata(clean, report.to_dict())
    PROJECT_PATHS.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 78)
    print("BITÁCORA DEL PIPELINE")
    print("-" * 78)
    for step, message in enumerate(report.steps, start=1):
        print(f"  {step:2d}. {message}")

    print("-" * 78)
    print("VERIFICACIÓN DE LOS REQUISITOS DEL DATASET (sección 7 de la consigna)")
    print("-" * 78)
    checks = [
        ("Al menos 500 registros", len(clean) >= 500, f"{len(clean):,} registros"),
        ("Al menos 6 variables", len(clean.columns) >= 6, f"{len(clean.columns)} variables"),
        (
            "Variables numéricas y categóricas",
            len(clean.select_dtypes(include=["number"]).columns) > 0
            and len(clean.select_dtypes(include=["category", "object", "bool"]).columns) > 0,
            f"{len(clean.select_dtypes(include=['number']).columns)} numéricas / "
            f"{len(clean.select_dtypes(include=['category', 'object', 'bool']).columns)} categóricas",
        ),
        ("Variable temporal", "fecha" in clean.columns, "columna 'fecha' (diaria)"),
        (
            "Variable geográfica",
            {"latitud", "longitud"}.issubset(clean.columns),
            "latitud, longitud, altitud y provincia",
        ),
        ("Sin valores ausentes", int(clean.isna().sum().sum()) == 0, f"{int(clean.isna().sum().sum())} ausentes"),
        ("Sin duplicados de clave", not clean.duplicated(["provincia", "fecha"]).any(), "clave única verificada"),
        ("Fuente verificable y citada", True, DATA_SOURCE.organization),
        ("Sin datos personales sensibles", True, "sólo variables ambientales agregadas por punto de malla"),
    ]
    for label, passed, detail in checks:
        print(f"  [{'OK' if passed else 'FALLA'}] {label:38s} → {detail}")

    print("-" * 78)
    print(f"Dataset limpio       : {PROJECT_PATHS.clean_dataset}")
    print(f"Diccionario de datos : {PROJECT_PATHS.data_dictionary}  ({len(dictionary)} variables)")
    print(f"Metadatos            : {PROJECT_PATHS.metadata}")
    print("=" * 78)
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
