"""Paso 3 del flujo de datos: cálculo reproducible de resultados analíticos.

Uso::

    python scripts/03_generar_resultados.py

Produce ``report/resultados.json``, fuente única de cifras para el dashboard,
el cuaderno y el informe técnico. Todas las técnicas emplean la semilla global
de :mod:`config.settings`, por lo que el resultado es reproducible.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Evita que joblib intente inferir la topología física en entornos Windows
# restringidos. No cambia los modelos ni la semilla; sólo elimina un aviso del
# backend al consultar información de hardware no expuesta por el sistema.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from config.settings import (  # noqa: E402
    MULTIVARIATE_FEATURES,
    NUMERIC_ANALYSIS_VARIABLES,
    PROJECT_PATHS,
    RANDOM_STATE,
)
from utils.metrics import (  # noqa: E402
    aggregate_annual,
    aggregate_by_province,
    aggregate_monthly,
    compare_sources,
    compute_kpis,
)
from utils.ml_models import (  # noqa: E402
    detect_anomalies,
    run_hierarchical_clustering,
    run_kmeans,
    run_pca,
    train_frost_classifier,
)
from utils.stats_tools import (  # noqa: E402
    compare_groups,
    fit_linear_trend,
    mann_kendall_test,
    top_correlations,
    variance_inflation_factors,
)

RESULTS_PATH = PROJECT_PATHS.report / "resultados.json"

HIERARCHICAL_FEATURES = [
    "heladas_por_anio",
    "tasa_helada",
    "tmin_media",
    "tmin_absoluta",
    "oscilacion_media",
    "precipitacion_anual",
    "humedad_media",
    "nubosidad_media",
    "radiacion_media",
    "viento_medio",
    "riesgo_medio",
    "altitud",
]


def _json_safe(value: Any) -> Any:
    """Convierte objetos científicos a tipos válidos para JSON estricto."""
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _trend_result(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Empaqueta OLS y Mann-Kendall para una serie ordenada."""
    linear = fit_linear_trend(x, y)
    mann_kendall = mann_kendall_test(y)
    return {
        "ols": (
            {
                **asdict(linear),
                "significativa": linear.is_significant,
                "direccion": linear.direction,
            }
            if linear is not None
            else None
        ),
        "mann_kendall": (
            {
                **asdict(mann_kendall),
                "significativa": mann_kendall.is_significant,
            }
            if mann_kendall is not None
            else None
        ),
    }


def _group_comparison_payload(frame: pd.DataFrame, group: str) -> dict[str, Any] | None:
    """Calcula y serializa un contraste de temperatura mínima entre grupos."""
    comparison = compare_groups(frame, "temperatura_minima", group)
    if comparison is None:
        return None
    return {
        "variable_respuesta": "temperatura_minima",
        "variable_grupo": group,
        **asdict(comparison),
        "significativa": comparison.is_significant,
    }


def build_results(frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    """Calcula el conjunto completo de resultados exigido por el proyecto."""
    province = aggregate_by_province(frame)
    monthly = aggregate_monthly(frame)
    annual = aggregate_annual(frame)

    trend = {
        "heladas_por_provincia": _trend_result(annual["anio"], annual["heladas_por_provincia"]),
        "temperatura_minima_media": _trend_result(annual["anio"], annual["tmin_media"]),
    }

    altitude_gradient = fit_linear_trend(province["altitud"] / 100.0, province["tmin_media"])
    gradient_payload = None
    if altitude_gradient is not None:
        gradient_payload = {
            "variable_respuesta": "temperatura_minima_media_provincial",
            "predictor": "altitud",
            "unidad_pendiente": "°C por 100 m",
            "pendiente_por_100_m": altitude_gradient.slope,
            "intercepto": altitude_gradient.intercept,
            "r_cuadrado": altitude_gradient.r_squared,
            "p_valor": altitude_gradient.p_value,
            "error_estandar": altitude_gradient.standard_error,
            "intervalo_confianza_95": altitude_gradient.confidence_interval,
            "n_provincias": altitude_gradient.n,
            "significativa": altitude_gradient.is_significant,
            "direccion": altitude_gradient.direction,
        }

    correlation_findings = top_correlations(
        frame,
        list(NUMERIC_ANALYSIS_VARIABLES),
        target="temperatura_minima",
        limit=12,
    )
    correlations = [
        {**asdict(finding), "direccion": finding.direction} for finding in correlation_findings
    ]
    vif = variance_inflation_factors(frame, list(MULTIVARIATE_FEATURES))

    pca_result = run_pca(
        frame,
        list(MULTIVARIATE_FEATURES),
        n_components=len(MULTIVARIATE_FEATURES),
    )
    pca_payload = None
    if pca_result is not None:
        components = [f"CP{index + 1}" for index in range(len(pca_result.explained_variance))]
        pca_payload = {
            "variables": pca_result.features,
            "varianza_por_componente": [
                {
                    "componente": component,
                    "proporcion": float(proportion),
                    "porcentaje": float(proportion * 100.0),
                }
                for component, proportion in zip(components, pca_result.explained_variance)
            ],
            "varianza_acumulada": [
                {
                    "componente": component,
                    "proporcion": float(proportion),
                    "porcentaje": float(proportion * 100.0),
                }
                for component, proportion in zip(components, pca_result.cumulative_variance)
            ],
            "n_componentes_90": pca_result.n_components_90,
            "cargas_cp1_cp3": pca_result.loadings.iloc[:, :3]
            .rename_axis("variable")
            .reset_index(),
            "interpretacion_cp1": pca_result.interpret_component(0),
            "interpretacion_cp2": pca_result.interpret_component(1),
        }

    clustering_result = run_kmeans(frame, list(MULTIVARIATE_FEATURES), max_clusters=4)
    clustering_payload = None
    if clustering_result is not None:
        clustering_payload = {
            "k": clustering_result.n_clusters,
            "silueta": clustering_result.silhouette,
            "calidad": clustering_result.quality,
            "inercia": clustering_result.inertia,
            "variables": clustering_result.features,
            "diagnostico_por_k": clustering_result.diagnostics,
            "centroides": clustering_result.centroids.rename_axis("grupo").reset_index(),
            "perfil_grupos": clustering_result.profile,
        }

    hierarchical_result = run_hierarchical_clustering(
        province,
        HIERARCHICAL_FEATURES,
        n_clusters=4,
    )
    hierarchical_payload = None
    if hierarchical_result is not None:
        membership = [
            {"provincia": name, "grupo": int(label)}
            for name, label in zip(hierarchical_result.province_names, hierarchical_result.labels)
        ]
        hierarchical_payload = {
            "metodo": "Ward",
            "k": hierarchical_result.n_clusters,
            "variables": [feature for feature in HIERARCHICAL_FEATURES if feature in province.columns],
            "pertenencia": membership,
            "matriz_enlace": hierarchical_result.linkage_matrix,
        }

    anomaly_result = detect_anomalies(
        frame,
        list(MULTIVARIATE_FEATURES),
        contamination=0.01,
        top=15,
    )
    anomaly_payload = None
    if anomaly_result is not None:
        anomaly_payload = {
            "numero": anomaly_result.n_anomalies,
            "contaminacion": anomaly_result.contamination,
            "variables": anomaly_result.features,
            "top_15": anomaly_result.extremes,
        }

    classifier_result = train_frost_classifier(frame, list(MULTIVARIATE_FEATURES))
    classifier_payload = None
    if classifier_result is not None:
        classifier_payload = {
            "objetivo": "helada",
            "auc_roc": classifier_result.roc_auc,
            "precision_media": classifier_result.average_precision,
            "exactitud": classifier_result.accuracy,
            "sensibilidad": classifier_result.recall,
            "precision": classifier_result.precision,
            "matriz_confusion": classifier_result.confusion,
            "n_train": classifier_result.n_train,
            "n_test": classifier_result.n_test,
            "prevalencia": classifier_result.positive_rate,
            "variables": classifier_result.features,
            "top_12_importancias": classifier_result.importances.head(12),
        }

    suspicious = frame.loc[
        frame.get("precipitacion_sospechosa", pd.Series(False, index=frame.index)).astype(bool),
        [column for column in ("fecha", "provincia", "precipitacion") if column in frame.columns],
    ]

    result = {
        "generado_en": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "semilla_aleatoria": RANDOM_STATE,
        "resumen": {
            "registros": int(len(frame)),
            "variables": int(len(frame.columns)),
            "provincias": int(frame["provincia"].nunique()),
            "dias_calendario": int(frame["fecha"].nunique()),
            "periodo_inicio": frame["fecha"].min(),
            "periodo_fin": frame["fecha"].max(),
            "dias_helada_meteorologica": int(frame["helada"].sum()),
            "tasa_helada_meteorologica": round(float(frame["helada"].mean() * 100.0), 4),
            "dias_helada_agronomica": int(frame["helada_agronomica"].sum()),
            "tasa_helada_agronomica": round(float(frame["helada_agronomica"].mean() * 100.0), 4),
            "temperatura_minima_media": round(float(frame["temperatura_minima"].mean()), 3),
            "temperatura_minima_absoluta": round(float(frame["temperatura_minima"].min()), 3),
            "memoria_mb": round(float(frame.memory_usage(deep=True).sum() / (1024**2)), 2),
        },
        "kpis": [asdict(kpi) for kpi in compute_kpis(frame)],
        "por_provincia": province,
        "mensual": monthly,
        "anual": annual,
        "tendencia": trend,
        "gradiente_altitudinal": gradient_payload,
        "correlaciones": correlations,
        "vif": vif,
        "pca": pca_payload,
        "clustering": clustering_payload,
        "jerarquico": hierarchical_payload,
        "anomalias": anomaly_payload,
        "clasificador": classifier_payload,
        "validacion_fuentes": compare_sources(frame),
        "contrastes": {
            group: _group_comparison_payload(frame, group)
            for group in ("provincia", "temporada", "piso_ecologico")
        },
        "calidad_datos": {
            "registros_precipitacion_marcados": int(len(suspicious)),
            "registros_marcados": suspicious,
            "ausentes": int(frame.isna().sum().sum()),
            "duplicados_clave": int(frame.duplicated(["provincia", "fecha"]).sum()),
            "bitacora": metadata.get("pipeline_etl", {}),
        },
    }
    return _json_safe(result)


def _print_summary(results: dict[str, Any], destination: Path) -> None:
    """Muestra las cifras principales sin depender de la interfaz gráfica."""
    summary = results["resumen"]
    trend = results["tendencia"]["heladas_por_provincia"]
    pca = results.get("pca") or {}
    clustering = results.get("clustering") or {}
    classifier = results.get("clasificador") or {}

    print("=" * 78)
    print("RESULTADOS ANALÍTICOS REPRODUCIBLES")
    print("=" * 78)
    print(f"Registros / variables     : {summary['registros']:,} / {summary['variables']}")
    print(f"Cobertura                 : {summary['periodo_inicio']} a {summary['periodo_fin']}")
    print(f"Helada meteorológica      : {summary['tasa_helada_meteorologica']:.2f} %")
    print(f"Helada agronómica         : {summary['tasa_helada_agronomica']:.2f} %")
    if trend.get("ols"):
        print(
            "Tendencia de heladas OLS : "
            f"{trend['ols']['slope']:+.3f} días/provincia/año "
            f"(p={trend['ols']['p_value']:.4f})"
        )
    if pca:
        print(f"Componentes para 90 %     : {pca['n_componentes_90']}")
    if clustering:
        print(f"K-Means                   : k={clustering['k']}, silueta={clustering['silueta']:.3f}")
    if classifier:
        print(f"Random Forest             : AUC ROC={classifier['auc_roc']:.3f}")
    print(f"Archivo generado          : {destination}")
    print("=" * 78)


def main() -> int:
    """Calcula y persiste el contrato de resultados del proyecto."""
    if not PROJECT_PATHS.clean_dataset.exists():
        print(f"ERROR: no se encontró {PROJECT_PATHS.clean_dataset}")
        print("Ejecute primero: python scripts/02_procesar_datos.py")
        return 1

    PROJECT_PATHS.report.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PROJECT_PATHS.clean_dataset, parse_dates=["fecha"], low_memory=False)
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype("float32")
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].astype("category")
    metadata = json.loads(PROJECT_PATHS.metadata.read_text(encoding="utf-8"))
    results = build_results(frame, metadata)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _print_summary(results, RESULTS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
