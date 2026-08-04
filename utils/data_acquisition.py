"""Adquisición del conjunto de datos desde dos reanálisis climáticos abiertos.

Arquitectura de fuentes
-----------------------
El proyecto integra **dos** productos independientes:

* **Fuente primaria — ERA5-Land / ERA5** (Copernicus, servida por Open-Meteo).
  Aporta las diecinueve variables observadas que alimentan el dashboard. Se
  eligió por su resolución de ≈ 9 km, la mínima que distingue entre capitales
  provinciales vecinas del altiplano.
* **Fuente secundaria — MERRA-2** (NASA POWER). Aporta un segundo estimado
  independiente de la temperatura mínima diaria, producido por otra institución
  y con otro modelo. No alimenta el análisis: sirve para verificar que el
  régimen de heladas detectado no es un artefacto de un producto concreto.

Por qué importó la resolución
-----------------------------
La primera versión de este módulo usó únicamente MERRA-2 (malla de ≈ 55 km). A
esa escala, Chucuito, El Collao y Yunguyo caen en la misma celda y reciben
series **idénticas**; lo mismo ocurre con Azángaro, Huancané y San Antonio de
Putina. Cualquier ranking provincial, agrupamiento o correlación con la altitud
habría sido un artefacto de la malla. Además, la elevación de celda asignaba
3 668 msnm a Sandia, cuya capital está a ≈ 2 170 msnm, borrando el único
gradiente altitudinal fuerte de la región. ERA5-Land resuelve ambos problemas.

Decisión de diseño para la trazabilidad
---------------------------------------
Los datasets originales se guardan **tal como llegan del servicio**: nombres
nativos de las variables, unidades nativas (viento en km/h, presión en hPa,
insolación en segundos) y fechas como cadena ISO. Toda normalización ocurre en
:mod:`utils.preprocessing`, de modo que el paso «datos originales → datos
limpios» sea reproducible y auditable por un tercero.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd

from config.settings import (
    DATA_SOURCE,
    DATA_SOURCE_SECONDARY,
    ERA5_DAILY_VARIABLES,
    ERA5_MODEL,
    LOCAL_TIMEZONE,
    MERRA2_PARAMETERS,
    PROJECT_PATHS,
    PROVINCES,
    STUDY_PERIOD,
    Province,
    StudyPeriod,
)

LOGGER = logging.getLogger(__name__)

MAX_RETRIES: int = 4
RETRY_BACKOFF_SECONDS: float = 4.0
REQUEST_TIMEOUT_SECONDS: int = 180

#: Pausa entre peticiones consecutivas. Los dos servicios son infraestructura
#: pública gratuita; espaciar las peticiones es una condición de uso responsable.
COURTESY_DELAY_SECONDS: float = 1.5

_USER_AGENT: str = "UNAP-SIS328-DashboardHeladasPuno/1.0 (proyecto academico)"


class DataAcquisitionError(RuntimeError):
    """Error irrecuperable durante la descarga de un conjunto de datos."""


def _request_json(url: str, *, description: str) -> dict[str, Any]:
    """Ejecuta una petición HTTP GET con reintentos y retroceso exponencial.

    Args:
        url: URL absoluta a consultar.
        description: Texto identificativo para los mensajes de registro.

    Returns:
        Cuerpo de la respuesta decodificado como diccionario.

    Raises:
        DataAcquisitionError: Si se agotan los reintentos disponibles.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            LOGGER.info("%s (intento %d/%d)", description, attempt, MAX_RETRIES)
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                LOGGER.warning("Fallo en %s: %s. Reintentando en %.0f s", description, error, delay)
                time.sleep(delay)

    raise DataAcquisitionError(f"No fue posible completar «{description}» tras {MAX_RETRIES} intentos: {last_error}")


# ---------------------------------------------------------------------------
# Fuente primaria: ERA5-Land / ERA5 vía Open-Meteo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Era5PointResponse:
    """Respuesta cruda de la fuente primaria para un punto geográfico.

    Attributes:
        province: Provincia consultada.
        payload: Cuerpo JSON completo devuelto por el servicio.
    """

    province: Province
    payload: dict[str, Any]

    @property
    def grid_elevation(self) -> float:
        """Elevación del punto de malla, en metros sobre el nivel del mar."""
        return float(self.payload["elevation"])

    @property
    def grid_latitude(self) -> float:
        """Latitud del punto de malla efectivamente empleado."""
        return float(self.payload["latitude"])

    @property
    def grid_longitude(self) -> float:
        """Longitud del punto de malla efectivamente empleado."""
        return float(self.payload["longitude"])

    @property
    def native_units(self) -> dict[str, str]:
        """Unidades nativas declaradas por el servicio para cada variable."""
        return dict(self.payload.get("daily_units", {}))


def build_era5_url(province: Province, period: StudyPeriod) -> str:
    """Construye la URL de consulta a la fuente primaria.

    Args:
        province: Provincia cuya capital define el punto de consulta.
        period: Ventana temporal solicitada.

    Returns:
        URL absoluta lista para ser invocada.
    """
    query = urllib.parse.urlencode(
        {
            "latitude": province.latitude,
            "longitude": province.longitude,
            "start_date": period.start.isoformat(),
            "end_date": period.end.isoformat(),
            "daily": ",".join(ERA5_DAILY_VARIABLES),
            "timezone": LOCAL_TIMEZONE,
            "models": ERA5_MODEL,
        }
    )
    return f"{DATA_SOURCE.endpoint}?{query}"


def fetch_era5_point(province: Province, period: StudyPeriod = STUDY_PERIOD) -> Era5PointResponse:
    """Descarga la serie diaria de una provincia desde la fuente primaria.

    Args:
        province: Provincia a consultar.
        period: Ventana temporal solicitada.

    Returns:
        Respuesta cruda encapsulada en :class:`Era5PointResponse`.

    Raises:
        DataAcquisitionError: Si la respuesta carece de la estructura esperada.
    """
    payload = _request_json(
        build_era5_url(province, period),
        description=f"ERA5-Land · {province.name} ({province.capital})",
    )
    if "daily" not in payload or "time" not in payload.get("daily", {}):
        raise DataAcquisitionError(
            f"Respuesta inesperada para {province.name}: falta el bloque 'daily.time'. "
            f"Detalle del servicio: {payload.get('reason', payload)}"
        )
    return Era5PointResponse(province=province, payload=payload)


def era5_response_to_frame(response: Era5PointResponse) -> pd.DataFrame:
    """Convierte una respuesta de la fuente primaria en un dataframe.

    Se conservan los nombres y las unidades nativas del servicio para que el
    dataset original refleje con exactitud lo entregado por la fuente.

    Args:
        response: Respuesta cruda de un punto geográfico.

    Returns:
        Dataframe con una fila por día, precedido por las columnas de
        identificación territorial.
    """
    frame = pd.DataFrame(response.payload["daily"])
    frame = frame.rename(columns={"time": "fecha_iso"})

    province = response.province
    identity = {
        "provincia": province.name,
        "capital": province.capital,
        "latitud_solicitada": province.latitude,
        "longitud_solicitada": province.longitude,
        "latitud_malla": response.grid_latitude,
        "longitud_malla": response.grid_longitude,
        "elevacion_malla_m": response.grid_elevation,
        "altitud_referencial_m": province.reference_altitude,
        "cuenca": province.basin,
        "zona_agroecologica": province.agroecological_zone,
        "fuente": "ERA5-Land/ERA5 (Copernicus)",
        "modelo": ERA5_MODEL,
    }
    for column, value in reversed(identity.items()):
        frame.insert(0, column, value)

    ordered = [*identity, "fecha_iso", *ERA5_DAILY_VARIABLES]
    return frame.loc[:, [column for column in ordered if column in frame.columns]]


# ---------------------------------------------------------------------------
# Fuente secundaria: MERRA-2 vía NASA POWER
# ---------------------------------------------------------------------------


def build_merra2_url(province: Province, period: StudyPeriod) -> str:
    """Construye la URL de consulta a la fuente secundaria.

    Args:
        province: Provincia cuya capital define el punto de consulta.
        period: Ventana temporal solicitada.

    Returns:
        URL absoluta lista para ser invocada.
    """
    query = urllib.parse.urlencode(
        {
            "parameters": ",".join(MERRA2_PARAMETERS),
            "community": "AG",
            "latitude": province.latitude,
            "longitude": province.longitude,
            "start": period.api_start,
            "end": period.api_end,
            "format": "JSON",
        }
    )
    return f"{DATA_SOURCE_SECONDARY.endpoint}?{query}"


def fetch_merra2_point(province: Province, period: StudyPeriod = STUDY_PERIOD) -> pd.DataFrame:
    """Descarga la serie de validación de una provincia desde MERRA-2.

    Args:
        province: Provincia a consultar.
        period: Ventana temporal solicitada.

    Returns:
        Dataframe con la fecha nativa ``YYYYMMDD`` y los parámetros solicitados.

    Raises:
        DataAcquisitionError: Si la respuesta carece de la estructura esperada.
    """
    payload = _request_json(
        build_merra2_url(province, period),
        description=f"MERRA-2 · {province.name} ({province.capital})",
    )
    parameters = payload.get("properties", {}).get("parameter")
    if not parameters:
        raise DataAcquisitionError(
            f"Respuesta inesperada de MERRA-2 para {province.name}: falta "
            f"'properties.parameter'. Mensajes: {payload.get('messages')}"
        )

    frame = pd.DataFrame(parameters)
    frame.index.name = "YYYYMMDD"
    frame = frame.reset_index()
    frame.insert(0, "provincia", province.name)
    return frame


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def download_primary_dataset(
    *, period: StudyPeriod = STUDY_PERIOD, save_raw: bool = True
) -> pd.DataFrame:
    """Descarga e integra las series primarias de las trece provincias.

    Args:
        period: Ventana temporal solicitada.
        save_raw: Si se persiste la respuesta JSON de cada punto en ``data/raw/``.

    Returns:
        Dataframe consolidado por provincia y día, con nombres y unidades nativas.
    """
    PROJECT_PATHS.ensure_directories()
    frames: list[pd.DataFrame] = []
    units: dict[str, str] = {}

    for index, province in enumerate(PROVINCES, start=1):
        response = fetch_era5_point(province, period)
        units = units or response.native_units

        if save_raw:
            slug = province.capital.lower().replace(" ", "_")
            raw_file = PROJECT_PATHS.raw_data / f"era5_{index:02d}_{slug}.json"
            raw_file.write_text(json.dumps(response.payload, ensure_ascii=False), encoding="utf-8")

        frame = era5_response_to_frame(response)
        frames.append(frame)
        LOGGER.info(
            "%s: %d días · elevación de malla %.0f msnm",
            province.name,
            len(frame),
            response.grid_elevation,
        )

        if index < len(PROVINCES):
            time.sleep(COURTESY_DELAY_SECONDS)

    dataset = pd.concat(frames, ignore_index=True)
    dataset.to_csv(PROJECT_PATHS.original_dataset, index=False, encoding="utf-8")

    # Las unidades nativas se documentan aparte: son el insumo que justifica las
    # conversiones aplicadas después en el pipeline de transformación.
    (PROJECT_PATHS.raw_data / "unidades_nativas_era5.json").write_text(
        json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOGGER.info("Dataset primario guardado en %s (%d filas)", PROJECT_PATHS.original_dataset, len(dataset))
    return dataset


def download_validation_dataset(*, period: StudyPeriod = STUDY_PERIOD) -> pd.DataFrame:
    """Descarga la serie secundaria empleada en la validación cruzada.

    Args:
        period: Ventana temporal solicitada.

    Returns:
        Dataframe consolidado con los parámetros de MERRA-2 por provincia y día.
    """
    PROJECT_PATHS.ensure_directories()
    frames: list[pd.DataFrame] = []

    for index, province in enumerate(PROVINCES, start=1):
        frames.append(fetch_merra2_point(province, period))
        if index < len(PROVINCES):
            time.sleep(COURTESY_DELAY_SECONDS)

    dataset = pd.concat(frames, ignore_index=True)
    destination = PROJECT_PATHS.data / "datos_originales_validacion_merra2.csv"
    dataset.to_csv(destination, index=False, encoding="utf-8")
    LOGGER.info("Dataset de validación guardado en %s (%d filas)", destination, len(dataset))
    return dataset
