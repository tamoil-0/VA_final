"""Verificación funcional del dataset y de la capa analítica.

Las pruebas cubren los requisitos mínimos de la sección 7 de la consigna y los
contratos que sostienen los indicadores, filtros y técnicas avanzadas.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from config.settings import PROJECT_PATHS
from utils.filters import apply_filters, build_default_state
from utils.metrics import aggregate_annual, aggregate_by_province, aggregate_monthly, compute_kpis
from utils.ml_models import run_kmeans, run_pca
from utils.stats_tools import fit_linear_trend, mann_kendall_test


@pytest.fixture(scope="session")
def clean_data() -> pd.DataFrame:
    """Carga una vez el dataset analítico usado por todas las pruebas."""
    return pd.read_csv(PROJECT_PATHS.clean_dataset, parse_dates=["fecha"], low_memory=False)


def test_dataset_meets_official_minimum_requirements(clean_data: pd.DataFrame) -> None:
    """El artefacto limpio satisface los siete requisitos verificables."""
    assert len(clean_data) >= 500
    assert clean_data.shape[1] >= 6
    assert len(clean_data.select_dtypes(include=["number"]).columns) > 0
    assert len(clean_data.select_dtypes(include=["object", "category", "bool"]).columns) > 0
    assert pd.api.types.is_datetime64_any_dtype(clean_data["fecha"])
    assert {"provincia", "latitud", "longitud", "altitud"}.issubset(clean_data.columns)
    assert clean_data.isna().sum().sum() == 0
    assert not clean_data.duplicated(["provincia", "fecha"]).any()
    assert PROJECT_PATHS.metadata.exists()
    assert PROJECT_PATHS.data_dictionary.exists()


def test_filters_respect_dates_and_never_add_rows(clean_data: pd.DataFrame) -> None:
    """El filtrado global respeta fechas, territorio y cardinalidad."""
    default = build_default_state(clean_data)
    state = replace(
        default,
        date_range=(date(2020, 1, 1), date(2020, 12, 31)),
        provinces=("Puno",),
    )
    filtered = apply_filters(clean_data, state)

    assert 0 < len(filtered) <= len(clean_data)
    assert filtered["fecha"].min().date() >= state.date_range[0]
    assert filtered["fecha"].max().date() <= state.date_range[1]
    assert set(filtered["provincia"].unique()) == {"Puno"}


def test_kpis_and_aggregations_are_available(clean_data: pd.DataFrame) -> None:
    """La capa de métricas devuelve más que el mínimo oficial de cuatro KPI."""
    kpis = compute_kpis(clean_data)
    assert len(kpis) >= 4
    assert all(kpi.key and kpi.label and kpi.value for kpi in kpis)
    assert not aggregate_by_province(clean_data).empty
    assert not aggregate_monthly(clean_data).empty
    assert not aggregate_annual(clean_data).empty


def test_trend_methods_detect_a_known_monotonic_signal() -> None:
    """OLS y Mann-Kendall reconocen una tendencia creciente inequívoca."""
    x = np.arange(1, 31, dtype="float64")
    y = 2.5 * x + np.sin(x) * 0.05

    linear = fit_linear_trend(x, y)
    mann_kendall = mann_kendall_test(y)

    assert linear is not None
    assert linear.slope == pytest.approx(2.5, rel=0.01)
    assert linear.is_significant
    assert linear.direction == "creciente"
    assert mann_kendall is not None
    assert mann_kendall.is_significant
    assert mann_kendall.trend == "creciente"
    assert mann_kendall.sen_slope > 0


def test_pca_and_kmeans_return_coherent_results(clean_data: pd.DataFrame) -> None:
    """Las técnicas multivariantes mantienen dimensiones y etiquetas válidas."""
    sample = clean_data.iloc[::100].copy()
    features = [
        "temperatura_minima",
        "oscilacion_termica",
        "precipitacion",
        "humedad_relativa",
        "radiacion_solar",
    ]

    pca = run_pca(sample, features, n_components=3)
    assert pca is not None
    assert pca.scores.shape == (len(sample), 3)
    assert pca.loadings.shape == (len(features), 3)
    assert np.all(np.diff(pca.cumulative_variance) >= 0)
    assert 0 < pca.cumulative_variance[-1] <= 1.0 + 1e-9

    clustering = run_kmeans(sample, features, n_clusters=3, max_clusters=3)
    assert clustering is not None
    assert clustering.n_clusters == 3
    assert len(clustering.labels) == len(sample)
    assert set(np.unique(clustering.labels)) == {0, 1, 2}
    assert -1.0 <= clustering.silhouette <= 1.0
    assert not clustering.profile.empty


def test_frost_risk_index_stays_in_declared_range(clean_data: pd.DataFrame) -> None:
    """El índice compuesto conserva su contrato fijo de 0 a 100."""
    risk = clean_data["indice_riesgo_helada"]
    assert risk.between(0.0, 100.0, inclusive="both").all()
