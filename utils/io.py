"""Carga de datos con caché y exportación de resultados.

Este módulo es el único punto del proyecto que lee del disco los artefactos de
datos. Las funciones de carga están decoradas con la caché de Streamlit, de modo
que el dataset se lee una sola vez por sesión y no en cada interacción del
usuario: sin ello, cada movimiento de un filtro releería 47 000 filas desde el
sistema de archivos.

El decorador se aplica de forma **tolerante**: si Streamlit no está disponible
—por ejemplo cuando el módulo se importa desde el cuaderno de análisis o desde
las pruebas— la función se comporta como una función corriente sin caché.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from config.settings import PROJECT_PATHS

LOGGER = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

#: Columnas que deben reinterpretarse como categóricas ordenadas al leer el CSV.
#: El formato CSV no preserva el orden de las categorías, y ese orden es
#: semántico: sin él, «Extrema» se ordenaría alfabéticamente antes de «Ligera».
_ORDERED_CATEGORIES: dict[str, str] = {
    "mes_nombre": "MONTH_LABELS",
    "mes_abrev": "MONTH_ABBREV",
    "trimestre": "QUARTERS",
    "temporada": "SEASON_ORDER",
    "piso_ecologico": "ECOLOGICAL_TIER_ORDER",
    "intensidad_helada": "FROST_INTENSITY_ORDER",
    "nivel_riesgo": "RISK_LEVEL_ORDER",
}


def _cache_data(**cache_kwargs: Any) -> Callable[[_F], _F]:
    """Devuelve el decorador de caché de Streamlit si está disponible.

    Args:
        **cache_kwargs: Argumentos que se pasan a ``st.cache_data``.

    Returns:
        El decorador de caché, o la identidad si Streamlit no está en ejecución.
    """

    def decorator(function: _F) -> _F:
        try:
            import streamlit as st
        except ImportError:  # pragma: no cover - ruta usada en cuadernos y pruebas
            return function
        return st.cache_data(**cache_kwargs)(function)  # type: ignore[return-value]

    return decorator


def _restore_ordered_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Restaura el orden semántico de las variables categóricas ordenadas.

    Args:
        frame: Dataframe recién leído desde CSV.

    Returns:
        Dataframe con las categorías ordenadas correctamente declaradas.
    """
    from config.settings import (
        ECOLOGICAL_TIER_ORDER,
        FROST_INTENSITY_ORDER,
        MONTH_ABBREV,
        MONTH_LABELS,
        RISK_LEVEL_ORDER,
        SEASON_ORDER,
    )

    orders: dict[str, tuple[str, ...]] = {
        "mes_nombre": tuple(MONTH_LABELS.values()),
        "mes_abrev": tuple(MONTH_ABBREV.values()),
        "trimestre": ("T1", "T2", "T3", "T4"),
        "temporada": SEASON_ORDER,
        "piso_ecologico": ECOLOGICAL_TIER_ORDER,
        "intensidad_helada": FROST_INTENSITY_ORDER,
        "nivel_riesgo": RISK_LEVEL_ORDER,
    }

    result = frame.copy()
    for column, categories in orders.items():
        if column not in result.columns:
            continue
        present = [category for category in categories if category in set(result[column].dropna().unique())]
        result[column] = pd.Categorical(result[column], categories=present, ordered=True)

    for column in ("provincia", "capital", "cuenca", "zona_agroecologica", "campania_agricola", "anio_mes", "fuente"):
        if column in result.columns:
            result[column] = result[column].astype("category")

    return result


@_cache_data(show_spinner="Cargando el conjunto de datos analítico…", ttl=None)
def load_clean_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el dataset analítico final.

    Args:
        path: Ruta alternativa al CSV limpio. ``None`` usa la ruta del proyecto.

    Returns:
        Dataframe listo para el análisis, con tipos y categorías restaurados.

    Raises:
        FileNotFoundError: Si el dataset no existe todavía, con la instrucción
            concreta para generarlo.
    """
    target = Path(path) if path is not None else PROJECT_PATHS.clean_dataset
    if not target.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset limpio en {target}.\n"
            "Genérelo ejecutando, en este orden:\n"
            "  python scripts/01_descargar_datos.py\n"
            "  python scripts/02_procesar_datos.py"
        )

    frame = pd.read_csv(target, parse_dates=["fecha"])
    frame = _restore_ordered_categories(frame)

    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype("float32")

    LOGGER.info("Dataset limpio cargado: %d filas × %d columnas", len(frame), len(frame.columns))
    return frame


@_cache_data(show_spinner=False, ttl=None)
def load_original_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el dataset original sin procesar.

    Args:
        path: Ruta alternativa al CSV original. ``None`` usa la ruta del proyecto.

    Returns:
        Dataframe con los datos tal como se descargaron del servicio.

    Raises:
        FileNotFoundError: Si el dataset original no existe.
    """
    target = Path(path) if path is not None else PROJECT_PATHS.original_dataset
    if not target.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset original en {target}. "
            "Ejecute: python scripts/01_descargar_datos.py"
        )
    return pd.read_csv(target)


@_cache_data(show_spinner=False, ttl=None)
def load_data_dictionary(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el diccionario de variables.

    Args:
        path: Ruta alternativa al CSV del diccionario.

    Returns:
        Dataframe con la documentación de cada variable. Si el archivo no existe
        se devuelve un dataframe vacío, para que la interfaz degrade con
        elegancia en lugar de interrumpirse.
    """
    target = Path(path) if path is not None else PROJECT_PATHS.data_dictionary
    if not target.exists():
        LOGGER.warning("Diccionario de datos no encontrado en %s", target)
        return pd.DataFrame()
    return pd.read_csv(target)


@_cache_data(show_spinner=False, ttl=None)
def load_metadata(path: str | Path | None = None) -> dict[str, Any]:
    """Carga los metadatos de trazabilidad del proceso ETL.

    Args:
        path: Ruta alternativa al JSON de metadatos.

    Returns:
        Diccionario de metadatos, o un diccionario vacío si el archivo no existe.
    """
    target = Path(path) if path is not None else PROJECT_PATHS.metadata
    if not target.exists():
        LOGGER.warning("Metadatos no encontrados en %s", target)
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------


def timestamped_filename(stem: str, extension: str) -> str:
    """Compone un nombre de archivo con marca temporal.

    Args:
        stem: Nombre base del archivo, sin extensión.
        extension: Extensión sin punto (``csv``, ``xlsx``).

    Returns:
        Nombre de archivo con el instante de generación incorporado, de modo que
        descargas sucesivas no se sobrescriban entre sí.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{stem}_{stamp}.{extension}"


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serializa un dataframe a CSV codificado en UTF-8 con BOM.

    La marca de orden de bytes es deliberada: sin ella, Microsoft Excel en
    configuración regional española interpreta el archivo como Latin-1 y muestra
    los acentos y los nombres de provincia corrompidos.

    Args:
        frame: Dataframe a exportar.

    Returns:
        Contenido del archivo CSV como secuencia de bytes.
    """
    return frame.to_csv(index=False).encode("utf-8-sig")


def dataframes_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Serializa varios dataframes a un único libro de Excel.

    Args:
        sheets: Diccionario ``{nombre_de_hoja: dataframe}``. Los nombres se
            recortan a 31 caracteres, límite que impone el formato XLSX.

    Returns:
        Contenido del libro de Excel como secuencia de bytes.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            export = frame.copy()
            # Excel no admite información de zona horaria en las celdas de fecha.
            for column in export.select_dtypes(include=["datetimetz"]).columns:
                export[column] = export[column].dt.tz_localize(None)
            export.to_excel(writer, sheet_name=name[:31], index=False)

            # Ajuste de ancho de columna: un libro exportado debe poder leerse
            # sin que el usuario tenga que redimensionar cada columna a mano.
            worksheet = writer.sheets[name[:31]]
            for position, column in enumerate(export.columns, start=1):
                header_width = len(str(column))
                sample = export[column].astype(str).head(200)
                content_width = int(sample.str.len().max()) if len(sample) else 0
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=position).column_letter
                ].width = min(42, max(11, header_width + 2, content_width + 2))
    return buffer.getvalue()


def figure_to_png_bytes(figure: Any, *, scale: int = 3) -> bytes | None:
    """Exporta una figura de Plotly a PNG de alta resolución.

    Args:
        figure: Figura de Plotly a exportar.
        scale: Factor de escala. Con ``3`` la imagen resulta adecuada para
            insertarse en un informe impreso.

    Returns:
        Los bytes del PNG, o ``None`` si el motor de exportación estática no
        está instalado. Devolver ``None`` en lugar de propagar la excepción
        permite que la interfaz oculte el botón de descarga sin romperse.
    """
    try:
        return figure.to_image(format="png", scale=scale)
    except Exception as error:  # pragma: no cover - depende del entorno
        LOGGER.warning("Exportación de imagen no disponible: %s", error)
        return None
