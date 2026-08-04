"""Pipeline de limpieza, transformación e integración del conjunto de datos.

El pipeline es **auditable**: cada etapa registra qué hizo y a cuántos registros
afectó en un objeto :class:`CleaningReport`, que después se serializa en
``data/metadata.json`` y se reproduce íntegro en el informe técnico. La
transición «datos originales → datos limpios» no es, por tanto, una caja negra.

Etapas
------
1.  Normalización del esquema: nombres nativos del servicio → nombres semánticos.
2.  Conversión de la fecha ISO a tipo temporal verdadero.
3.  Homogeneización de unidades (km/h→m/s, hPa→kPa, s→h), documentada.
4.  Sustitución de centinelas de ausencia por valores nulos.
5.  Validación de rangos físicos y anulación de observaciones imposibles.
6.  Verificación de coherencia interna (mínima ≤ media ≤ máxima).
7.  Marcado —sin destruir— de extremos de precipitación implausibles.
8.  Control de unicidad sobre la clave natural (provincia, fecha).
9.  Reindexado al calendario completo, para materializar días ausentes.
10. Imputación temporal por interpolación dentro de cada provincia.
11. **Integración** de la fuente secundaria de validación cruzada.
12. Generación de variables derivadas de dominio agroclimático.
13. Ordenamiento de categorías y tipado final eficiente en memoria.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    AGRICULTURAL_CAMPAIGN_START_MONTH,
    API_FILL_VALUE,
    ECOLOGICAL_TIER_ORDER,
    ERA5_DAILY_VARIABLES,
    FROST_INTENSITY_BINS,
    FROST_INTENSITY_ORDER,
    FROST_THRESHOLDS,
    MERRA2_PARAMETERS,
    MONTH_ABBREV,
    MONTH_LABELS,
    RISK_LEVEL_ORDER,
    SEASON_BY_MONTH,
    SEASON_ORDER,
    UNIT_CONVERSIONS,
    VARIABLE_DICTIONARY,
    classify_ecological_tier,
)

LOGGER = logging.getLogger(__name__)

#: Rangos físicamente admisibles por variable, expresados en las unidades
#: **finales** del dataset (es decir, después de la homogeneización).
#:
#: Los límites se fijaron tras verificar las unidades reales que devuelve el
#: servicio, no las que sugiere el nombre del parámetro. Una calibración
#: descuidada de estos umbrales es especialmente peligrosa: anularía series
#: completas que después la imputación rellenaría por interpolación, corrompiendo
#: el análisis multivariante sin dejar ningún rastro visible.
PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "temperatura_media": (-30.0, 35.0),
    "temperatura_maxima": (-25.0, 45.0),
    "temperatura_minima": (-40.0, 30.0),
    "punto_rocio": (-45.0, 30.0),
    "temperatura_suelo": (-30.0, 45.0),
    "humedad_relativa": (0.0, 100.0),
    "nubosidad": (0.0, 100.0),
    # Umbral holgado a propósito: por encima de 400 mm/día la cifra sería
    # imposible, pero los extremos altos aunque implausibles NO se anulan aquí;
    # se marcan en `_flag_precipitation_outliers` para que sigan siendo visibles
    # como limitación del dato en lugar de desaparecer por interpolación.
    "precipitacion": (0.0, 400.0),
    "lluvia": (0.0, 400.0),
    "nevada": (0.0, 200.0),
    "horas_precipitacion": (0.0, 24.0),
    "radiacion_solar": (0.0, 45.0),
    "horas_sol": (0.0, 24.0),
    "velocidad_viento": (0.0, 40.0),
    "racha_viento_maxima": (0.0, 70.0),
    "presion_superficie": (40.0, 110.0),
    "evapotranspiracion": (0.0, 20.0),
    "deficit_presion_vapor_maximo": (0.0, 8.0),
    "humedad_suelo": (0.0, 1.0),
}

#: Umbral de precipitación diaria (mm) por encima del cual el registro se
#: considera sospechoso para el altiplano peruano. Las estaciones de superficie
#: del SENAMHI no reportan acumulados diarios de esta magnitud en la cuenca del
#: Titicaca; valores superiores provienen de la parametrización convectiva del
#: reanálisis sobre topografía compleja.
SUSPICIOUS_PRECIPITATION_MM: float = 100.0

#: Columnas de identidad territorial: no dependen del día y deben repoblarse
#: íntegras después de cualquier reindexado.
TERRITORY_COLUMNS: tuple[str, ...] = (
    "capital",
    "latitud",
    "longitud",
    "altitud",
    "cuenca",
    "zona_agroecologica",
    "fuente",
)


@dataclass
class CleaningReport:
    """Bitácora de todas las operaciones del pipeline.

    Attributes:
        rows_input: Filas del dataset original.
        rows_output: Filas del dataset limpio.
        columns_input: Columnas del dataset original.
        columns_output: Columnas del dataset limpio.
        unit_conversions: Conversiones de unidad aplicadas, por variable.
        sentinel_replaced: Centinelas de ausencia sustituidos, por variable.
        out_of_range: Observaciones anuladas por rango físico, por variable.
        inconsistent_temperatures: Filas con orden térmico incoherente corregidas.
        suspicious_precipitation: Acumulados de lluvia marcados como implausibles.
        suspicious_precipitation_dates: Fechas dominantes de esos episodios y
            número de provincias afectadas en cada una.
        duplicates_removed: Filas duplicadas eliminadas.
        missing_dates_added: Días ausentes del calendario que se reindexaron.
        imputed: Valores imputados por interpolación, por variable.
        residual_missing: Valores ausentes que persisten tras la imputación.
        validation_source_merged: Filas que recibieron la serie de validación.
        validation_agreement: Métricas de concordancia entre ambas fuentes.
        derived_variables: Variables generadas en la fase de transformación.
        steps: Descripción ordenada de las etapas ejecutadas.
    """

    rows_input: int = 0
    rows_output: int = 0
    columns_input: int = 0
    columns_output: int = 0
    unit_conversions: dict[str, str] = field(default_factory=dict)
    sentinel_replaced: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    inconsistent_temperatures: int = 0
    suspicious_precipitation: int = 0
    suspicious_precipitation_dates: dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    missing_dates_added: int = 0
    imputed: dict[str, int] = field(default_factory=dict)
    residual_missing: dict[str, int] = field(default_factory=dict)
    validation_source_merged: int = 0
    validation_agreement: dict[str, float] = field(default_factory=dict)
    derived_variables: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Añade una etapa a la bitácora y la envía al sistema de registro.

        Args:
            message: Descripción de la operación realizada.
        """
        self.steps.append(message)
        LOGGER.info(message)

    @property
    def total_sentinels(self) -> int:
        """Total de centinelas sustituidos en todas las variables."""
        return int(sum(self.sentinel_replaced.values()))

    @property
    def total_out_of_range(self) -> int:
        """Total de observaciones anuladas por rango físico."""
        return int(sum(self.out_of_range.values()))

    @property
    def total_imputed(self) -> int:
        """Total de valores imputados en todas las variables."""
        return int(sum(self.imputed.values()))

    def to_dict(self) -> dict[str, Any]:
        """Serializa la bitácora a un diccionario apto para JSON."""
        payload = asdict(self)
        payload["total_sentinels"] = self.total_sentinels
        payload["total_out_of_range"] = self.total_out_of_range
        payload["total_imputed"] = self.total_imputed
        return payload


# ---------------------------------------------------------------------------
# Etapas de limpieza
# ---------------------------------------------------------------------------


def _normalize_schema(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Renombra las columnas nativas del servicio y fija el esquema de trabajo.

    Args:
        frame: Dataset original tal como se descargó.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con nombres de columna semánticos en español.
    """
    result = frame.rename(columns=dict(ERA5_DAILY_VARIABLES))
    result = result.rename(
        columns={
            "latitud_malla": "latitud",
            "longitud_malla": "longitud",
            "elevacion_malla_m": "altitud",
        }
    )
    result = result.drop(
        columns=[c for c in ("latitud_solicitada", "longitud_solicitada", "modelo") if c in result.columns]
    )
    report.log(
        f"Normalización del esquema: {len(ERA5_DAILY_VARIABLES)} variables nativas del "
        "servicio se renombraron a nombres semánticos en español y se unificó la "
        "nomenclatura geográfica (latitud, longitud, altitud)."
    )
    return result


def _parse_dates(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Convierte la fecha ISO en un tipo temporal verdadero.

    Args:
        frame: Dataframe con la columna ``fecha_iso``.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con la columna ``fecha`` de tipo ``datetime64[ns]``.

    Raises:
        ValueError: Si alguna fecha no pudo interpretarse.
    """
    result = frame.copy()
    result["fecha"] = pd.to_datetime(result["fecha_iso"], format="%Y-%m-%d", errors="coerce")

    unparsed = int(result["fecha"].isna().sum())
    if unparsed:
        raise ValueError(f"{unparsed} fechas no pudieron interpretarse con el formato ISO 'YYYY-MM-DD'.")

    result = result.drop(columns=["fecha_iso"])
    report.log(
        "La fecha llegaba como cadena de texto ISO; se convirtió a tipo temporal para "
        "habilitar filtros por rango, agregaciones cronológicas y cálculo de tendencias."
    )
    return result


def _harmonize_units(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Convierte las variables a las unidades convencionales en agrometeorología.

    El servicio entrega el viento en km/h, la presión en hPa y la insolación en
    segundos. Analizar sin convertir produciría interpretaciones erróneas y haría
    incomparables los resultados con la literatura del sector.

    Args:
        frame: Dataframe con las variables en unidades nativas.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con las unidades homogeneizadas.
    """
    result = frame.copy()
    for variable, (factor, source_unit, target_unit) in UNIT_CONVERSIONS.items():
        if variable not in result.columns:
            continue
        result[variable] = result[variable] * factor
        report.unit_conversions[variable] = f"{source_unit} → {target_unit} (×{factor:.6g})"

    report.log(
        f"Homogeneización de unidades en {len(report.unit_conversions)} variables: "
        + "; ".join(f"{name} {change}" for name, change in report.unit_conversions.items())
        + "."
    )
    return result


def _replace_sentinels(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Sustituye valores centinela de ausencia por nulos.

    Args:
        frame: Dataframe con las variables climáticas.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con ``NaN`` en lugar del centinela.
    """
    result = frame.copy()
    for variable in ERA5_DAILY_VARIABLES.values():
        if variable not in result.columns or not pd.api.types.is_numeric_dtype(result[variable]):
            continue
        mask = np.isclose(result[variable].to_numpy(dtype="float64"), API_FILL_VALUE, atol=0.5)
        count = int(mask.sum())
        if count:
            result.loc[mask, variable] = np.nan
            report.sentinel_replaced[variable] = count

    if report.total_sentinels:
        report.log(
            f"Se sustituyeron {report.total_sentinels:,} valores centinela ({API_FILL_VALUE}) "
            f"por ausentes en {len(report.sentinel_replaced)} variables."
        )
    else:
        report.log(
            f"No se hallaron valores centinela ({API_FILL_VALUE}); el servicio codifica la "
            "ausencia como nulo JSON, ya interpretado como ausente en la carga."
        )
    return result


def _enforce_physical_ranges(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Anula observaciones fuera del rango físicamente posible.

    Args:
        frame: Dataframe con las variables climáticas.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con las observaciones imposibles convertidas en ausentes.
    """
    result = frame.copy()
    for variable, (minimum, maximum) in PHYSICAL_RANGES.items():
        if variable not in result.columns:
            continue
        mask = result[variable].notna() & ((result[variable] < minimum) | (result[variable] > maximum))
        count = int(mask.sum())
        if count:
            result.loc[mask, variable] = np.nan
            report.out_of_range[variable] = count

    report.log(
        f"Validación de rangos físicos sobre {len(PHYSICAL_RANGES)} variables: "
        f"{report.total_out_of_range:,} observaciones fuera de los límites admisibles fueron anuladas."
    )
    return result


def _enforce_temperature_consistency(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Corrige filas donde el orden mínima ≤ media ≤ máxima no se cumple.

    Args:
        frame: Dataframe con las tres temperaturas diarias.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con la media recalculada donde había incoherencia.
    """
    result = frame.copy()
    columns = ["temperatura_minima", "temperatura_media", "temperatura_maxima"]
    if not all(column in result.columns for column in columns):
        return result

    complete = result[columns].notna().all(axis=1)
    inconsistent = complete & (
        (result["temperatura_minima"] > result["temperatura_media"])
        | (result["temperatura_media"] > result["temperatura_maxima"])
    )
    count = int(inconsistent.sum())

    if count:
        # Se conservan los extremos observados y se recalcula la media como su
        # punto medio: es la corrección de mínima intervención.
        result.loc[inconsistent, "temperatura_media"] = (
            result.loc[inconsistent, "temperatura_minima"] + result.loc[inconsistent, "temperatura_maxima"]
        ) / 2.0

    report.inconsistent_temperatures = count
    report.log(
        f"Verificación de coherencia térmica (mínima ≤ media ≤ máxima): {count:,} filas "
        "incoherentes; en ellas la media se recalculó como punto medio de los extremos observados."
    )
    return result


def _flag_precipitation_outliers(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Marca —sin eliminar— los acumulados diarios de lluvia implausibles.

    Se prefiere **marcar** antes que imputar por una razón metodológica: estos
    valores no son ruido aleatorio sino un sesgo conocido de los reanálisis sobre
    topografía compleja, y suelen aparecer de forma coherente en varias
    provincias el mismo día. Interpolarlos los haría desaparecer del análisis y
    con ellos la evidencia de una limitación real del conjunto de datos.

    Args:
        frame: Dataframe con la variable ``precipitacion``.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con la columna booleana ``precipitacion_sospechosa``.
    """
    result = frame.copy()
    suspicious = result["precipitacion"] > SUSPICIOUS_PRECIPITATION_MM
    result["precipitacion_sospechosa"] = suspicious.fillna(False)

    count = int(result["precipitacion_sospechosa"].sum())
    report.suspicious_precipitation = count

    if count:
        by_date = (
            result.loc[result["precipitacion_sospechosa"]]
            .groupby(result.loc[result["precipitacion_sospechosa"], "fecha"].dt.date, observed=True)
            .size()
            .sort_values(ascending=False)
        )
        report.suspicious_precipitation_dates = {str(date): int(n) for date, n in by_date.head(5).items()}
        worst_date, worst_n = next(iter(by_date.items()))
        report.log(
            f"Control de calidad de la precipitación: {count} acumulados diarios superiores a "
            f"{SUSPICIOUS_PRECIPITATION_MM:.0f} mm se marcaron como sospechosos y se conservaron sin "
            f"modificar. El episodio dominante es {worst_date}, con {worst_n} provincia(s) afectada(s) "
            "el mismo día; el dato permanece disponible y trazable para el análisis de anomalías."
        )
    else:
        report.log("Control de calidad de la precipitación: no se detectaron acumulados implausibles.")
    return result


def _drop_duplicates(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Elimina filas duplicadas según la clave natural (provincia, fecha).

    Args:
        frame: Dataframe a depurar.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe sin duplicados.
    """
    before = len(frame)
    result = frame.drop_duplicates(subset=["provincia", "fecha"], keep="first").reset_index(drop=True)
    report.duplicates_removed = before - len(result)
    report.log(
        f"Control de unicidad sobre la clave natural (provincia, fecha): "
        f"{report.duplicates_removed:,} registros duplicados eliminados."
    )
    return result


def _reindex_calendar(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Completa el calendario diario de cada provincia.

    Un día ausente es información: si no se materializa como fila, una media
    mensual se calcularía sobre un denominador incorrecto y un conteo de heladas
    subestimaría el fenómeno sin advertirlo.

    Args:
        frame: Dataframe con la serie por provincia.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con el calendario completo por provincia.
    """
    calendar = pd.date_range(frame["fecha"].min(), frame["fecha"].max(), freq="D")
    identity = (
        frame.loc[:, ["provincia", *[c for c in TERRITORY_COLUMNS if c in frame.columns]]]
        .drop_duplicates(subset=["provincia"])
        .set_index("provincia")
    )

    index = pd.MultiIndex.from_product([identity.index, calendar], names=["provincia", "fecha"])
    result = frame.set_index(["provincia", "fecha"]).reindex(index).reset_index()

    for column in identity.columns:
        result[column] = result["provincia"].map(identity[column])
    result["precipitacion_sospechosa"] = result["precipitacion_sospechosa"].fillna(False).astype(bool)

    report.missing_dates_added = len(result) - len(frame)
    report.log(
        f"Reindexado al calendario diario completo ({len(calendar):,} días × {len(identity)} "
        f"provincias = {len(calendar) * len(identity):,} filas esperadas): "
        f"{report.missing_dates_added:,} días ausentes materializados."
    )
    return result


def _impute_missing(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Imputa ausentes por interpolación temporal dentro de cada provincia.

    Se interpola en el tiempo —no con la media global— porque las variables
    climáticas presentan fuerte autocorrelación diaria: el vecino temporal es el
    mejor estimador disponible y preserva la estacionalidad, que una media global
    destruiría. Las variables de precipitación se tratan aparte: un ausente en una
    serie de lluvia se interpreta como ausencia de registro de lluvia y se fija en
    cero, evitando inventar precipitación inexistente.

    Args:
        frame: Dataframe con el calendario completo.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe sin valores ausentes en las variables continuas.
    """
    result = frame.sort_values(["provincia", "fecha"]).reset_index(drop=True)

    zero_filled = {"precipitacion", "lluvia", "nevada", "horas_precipitacion"}
    continuous = [
        variable
        for variable in ERA5_DAILY_VARIABLES.values()
        if variable in result.columns and variable not in zero_filled
    ]

    for variable in continuous:
        missing_before = int(result[variable].isna().sum())
        if not missing_before:
            continue
        result[variable] = result.groupby("provincia", observed=True)[variable].transform(
            lambda series: series.interpolate(method="linear", limit_direction="both")
        )
        filled = missing_before - int(result[variable].isna().sum())
        if filled:
            report.imputed[variable] = filled

    for variable in zero_filled & set(result.columns):
        missing_before = int(result[variable].isna().sum())
        if missing_before:
            result[variable] = result[variable].fillna(0.0)
            report.imputed[variable] = missing_before

    for variable in ERA5_DAILY_VARIABLES.values():
        if variable in result.columns:
            residual = int(result[variable].isna().sum())
            if residual:
                report.residual_missing[variable] = residual

    report.log(
        f"Imputación por provincia (interpolación lineal bidireccional en variables continuas; "
        f"relleno con cero en las de precipitación): {report.total_imputed:,} valores completados; "
        f"{sum(report.residual_missing.values()):,} ausentes residuales."
    )
    return result


def _merge_validation_source(
    frame: pd.DataFrame, validation: pd.DataFrame | None, report: CleaningReport
) -> pd.DataFrame:
    """Integra la fuente secundaria y cuantifica la concordancia entre ambas.

    Esta es la etapa de **integración** del pipeline: dos productos de reanálisis
    de instituciones distintas, con mallas y modelos distintos, se unen por la
    clave (provincia, fecha). La concordancia resultante es la evidencia de que la
    señal analizada no es un artefacto de un producto concreto.

    Args:
        frame: Dataset primario ya limpio.
        validation: Dataset secundario crudo, o ``None`` para omitir la etapa.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con las columnas de validación incorporadas.
    """
    if validation is None or validation.empty:
        report.log(
            "Integración de la fuente secundaria omitida: no se encontró el dataset de "
            "validación MERRA-2. El dashboard opera con la fuente primaria únicamente."
        )
        return frame

    secondary = validation.rename(columns=dict(MERRA2_PARAMETERS)).copy()
    secondary["fecha"] = pd.to_datetime(secondary["YYYYMMDD"].astype(str), format="%Y%m%d")
    secondary = secondary.drop(columns=["YYYYMMDD"])

    for column in MERRA2_PARAMETERS.values():
        if column in secondary.columns:
            secondary.loc[
                np.isclose(secondary[column].to_numpy(dtype="float64"), API_FILL_VALUE, atol=0.5), column
            ] = np.nan

    keep = ["provincia", "fecha", "temperatura_minima_merra2"]
    result = frame.merge(secondary.loc[:, keep], on=["provincia", "fecha"], how="left")
    report.validation_source_merged = int(result["temperatura_minima_merra2"].notna().sum())

    both = result[["temperatura_minima", "temperatura_minima_merra2"]].dropna()
    if len(both) > 2:
        primary_frost = both["temperatura_minima"] <= FROST_THRESHOLDS["meteorologica"]
        secondary_frost = both["temperatura_minima_merra2"] <= FROST_THRESHOLDS["meteorologica"]
        report.validation_agreement = {
            "n_pares": int(len(both)),
            "correlacion_pearson": round(float(both.corr().iloc[0, 1]), 4),
            "sesgo_medio_c": round(float((both.iloc[:, 0] - both.iloc[:, 1]).mean()), 3),
            "error_absoluto_medio_c": round(float((both.iloc[:, 0] - both.iloc[:, 1]).abs().mean()), 3),
            "concordancia_clasificacion_helada_%": round(float((primary_frost == secondary_frost).mean() * 100), 2),
        }

    agreement = report.validation_agreement
    report.log(
        "Integración de la fuente secundaria (MERRA-2 / NASA POWER) por la clave "
        f"(provincia, fecha): {report.validation_source_merged:,} filas emparejadas. "
        f"Correlación de la temperatura mínima entre fuentes r = "
        f"{agreement.get('correlacion_pearson', float('nan')):.3f}; sesgo medio "
        f"{agreement.get('sesgo_medio_c', float('nan')):+.2f} °C; concordancia en la "
        f"clasificación binaria de helada {agreement.get('concordancia_clasificacion_helada_%', float('nan')):.1f} %."
    )
    return result


# ---------------------------------------------------------------------------
# Etapa de transformación: variables derivadas
# ---------------------------------------------------------------------------


def _saturation_vapour_pressure(temperature_c: pd.Series) -> pd.Series:
    """Calcula la presión de vapor de saturación con la ecuación de Tetens.

    Args:
        temperature_c: Temperatura del aire en grados Celsius.

    Returns:
        Presión de vapor de saturación en kilopascales.
    """
    return 0.6108 * np.exp((17.27 * temperature_c) / (temperature_c + 237.3))


def _classify_frost_intensity(minimum_temperature: pd.Series) -> pd.Categorical:
    """Clasifica la severidad de la helada a partir de la temperatura mínima.

    Args:
        minimum_temperature: Serie de temperaturas mínimas diarias en °C.

    Returns:
        Variable categórica ordenada con la intensidad del evento.
    """
    labels = pd.Series("Sin helada", index=minimum_temperature.index, dtype=object)
    for label, lower, upper in FROST_INTENSITY_BINS:
        if label == "Sin helada":
            continue
        labels.loc[(minimum_temperature > lower) & (minimum_temperature <= upper)] = label
    return pd.Categorical(labels, categories=list(FROST_INTENSITY_ORDER), ordered=True)


def _agricultural_campaign(dates: pd.Series) -> pd.Series:
    """Asigna la campaña agrícola (agosto–julio) a cada fecha.

    Args:
        dates: Serie de fechas.

    Returns:
        Serie de texto con la campaña en formato ``AAAA-AAAA``.
    """
    start_year = np.where(
        dates.dt.month >= AGRICULTURAL_CAMPAIGN_START_MONTH, dates.dt.year, dates.dt.year - 1
    )
    return pd.Series([f"{year}-{year + 1}" for year in start_year], index=dates.index)


def _frost_risk_index(frame: pd.DataFrame) -> pd.Series:
    """Construye un índice compuesto de riesgo diario de helada (0–100).

    El índice integra los cuatro mecanismos físicos que la literatura
    agrometeorológica reconoce como determinantes de una **helada de radiación**,
    el tipo dominante en el altiplano:

    * **Déficit térmico** (peso 0.45): distancia por debajo del umbral
      agronómico; es el componente dominante y el único de efecto directo.
    * **Cielo despejado** (peso 0.25): la ausencia de nubosidad permite la
      pérdida radiativa nocturna hacia el espacio, condición necesaria del
      fenómeno. Se emplea la nubosidad observada, no una aproximación.
    * **Sequedad atmosférica** (peso 0.20): a menor humedad, menor inercia
      térmica y menor liberación de calor latente por condensación.
    * **Calma de viento** (peso 0.10): sin mezcla mecánica, el aire frío se
      estratifica junto al suelo, donde está el cultivo.

    Cada componente se normaliza al intervalo [0, 1] con límites **fijos** —no
    percentiles de la muestra— para que el índice sea comparable entre
    subconjuntos filtrados y no cambie de escala al mover un filtro. Ésta es una
    propiedad deliberada: un índice basado en percentiles daría valores distintos
    para el mismo día según lo que el usuario tuviera seleccionado.

    Args:
        frame: Dataframe con las variables climáticas ya limpias.

    Returns:
        Serie con el índice de riesgo en el rango 0–100.
    """
    threshold = FROST_THRESHOLDS["agronomica"]

    thermal_deficit = ((threshold - frame["temperatura_minima"]) / 12.0).clip(0.0, 1.0)
    clear_sky = (1.0 - frame["nubosidad"] / 100.0).clip(0.0, 1.0)
    dryness = ((100.0 - frame["humedad_relativa"]) / 60.0).clip(0.0, 1.0)
    calm = (1.0 - frame["velocidad_viento"] / 8.0).clip(0.0, 1.0)

    index = 0.45 * thermal_deficit + 0.25 * clear_sky + 0.20 * dryness + 0.10 * calm
    return (index * 100.0).round(2)


def add_derived_variables(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Genera todas las variables derivadas del modelo analítico.

    Args:
        frame: Dataframe limpio con las variables de origen.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe enriquecido con las variables temporales, territoriales y
        agroclimáticas que el dashboard consume.
    """
    result = frame.copy()

    # --- Descomposición temporal ---
    result["anio"] = result["fecha"].dt.year.astype("int16")
    result["mes"] = result["fecha"].dt.month.astype("int8")
    result["mes_nombre"] = pd.Categorical(
        result["mes"].map(MONTH_LABELS), categories=list(MONTH_LABELS.values()), ordered=True
    )
    result["mes_abrev"] = pd.Categorical(
        result["mes"].map(MONTH_ABBREV), categories=list(MONTH_ABBREV.values()), ordered=True
    )
    result["trimestre"] = pd.Categorical(
        "T" + result["fecha"].dt.quarter.astype(str), categories=["T1", "T2", "T3", "T4"], ordered=True
    )
    result["dia_anio"] = result["fecha"].dt.dayofyear.astype("int16")
    result["semana_anio"] = result["fecha"].dt.isocalendar().week.to_numpy().astype("int8")
    result["anio_mes"] = result["fecha"].dt.to_period("M").astype(str)
    result["temporada"] = pd.Categorical(
        result["mes"].map(SEASON_BY_MONTH), categories=list(SEASON_ORDER), ordered=True
    )
    result["campania_agricola"] = _agricultural_campaign(result["fecha"])

    # --- Caracterización territorial ---
    result["piso_ecologico"] = pd.Categorical(
        result["altitud"].apply(classify_ecological_tier),
        categories=list(ECOLOGICAL_TIER_ORDER),
        ordered=True,
    )

    # --- Variables agroclimáticas ---
    result["oscilacion_termica"] = (result["temperatura_maxima"] - result["temperatura_minima"]).round(2)
    result["helada"] = result["temperatura_minima"] <= FROST_THRESHOLDS["meteorologica"]
    result["helada_agronomica"] = result["temperatura_minima"] <= FROST_THRESHOLDS["agronomica"]
    result["intensidad_helada"] = _classify_frost_intensity(result["temperatura_minima"])
    result["deficit_termico"] = (
        (FROST_THRESHOLDS["agronomica"] - result["temperatura_minima"]).clip(lower=0.0).round(2)
    )
    result["dia_lluvia"] = result["precipitacion"] >= 1.0
    result["dia_nevada"] = result["nevada"] > 0.0

    # Déficit medio de presión de vapor: mide la sequedad real del aire. Se
    # calcula a partir de la temperatura media y el punto de rocío, y complementa
    # —no duplica— el máximo diario que entrega la fuente.
    result["deficit_presion_vapor"] = (
        (
            _saturation_vapour_pressure(result["temperatura_media"])
            - _saturation_vapour_pressure(result["punto_rocio"])
        )
        .clip(lower=0.0)
        .round(4)
    )

    result["indice_riesgo_helada"] = _frost_risk_index(result)
    result["nivel_riesgo"] = pd.Categorical(
        pd.cut(
            result["indice_riesgo_helada"],
            bins=[-0.01, 25.0, 40.0, 55.0, 70.0, 100.01],
            labels=list(RISK_LEVEL_ORDER),
        ),
        categories=list(RISK_LEVEL_ORDER),
        ordered=True,
    )

    if "temperatura_minima_merra2" in result.columns:
        result["helada_merra2"] = result["temperatura_minima_merra2"] <= FROST_THRESHOLDS["meteorologica"]

    derived = [
        "anio", "mes", "mes_nombre", "mes_abrev", "trimestre", "dia_anio", "semana_anio",
        "anio_mes", "temporada", "campania_agricola", "piso_ecologico", "oscilacion_termica",
        "helada", "helada_agronomica", "intensidad_helada", "deficit_termico", "dia_lluvia",
        "dia_nevada", "deficit_presion_vapor", "indice_riesgo_helada", "nivel_riesgo",
        "precipitacion_sospechosa",
    ]
    if "helada_merra2" in result.columns:
        derived.append("helada_merra2")

    report.derived_variables = derived
    report.log(
        f"Transformación: se generaron {len(derived)} variables derivadas "
        "(descomposición temporal, caracterización territorial por piso ecológico, "
        "indicadores de helada e índice compuesto de riesgo agroclimático)."
    )
    return result


def _optimize_dtypes(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Ajusta los tipos de datos para reducir la huella de memoria.

    Args:
        frame: Dataframe final.
        report: Bitácora que se actualiza con la operación.

    Returns:
        Dataframe con tipos compactos y categorías declaradas.
    """
    result = frame.copy()

    for column in ("provincia", "capital", "cuenca", "zona_agroecologica", "campania_agricola", "anio_mes", "fuente"):
        if column in result.columns:
            result[column] = result[column].astype("category")

    for column in result.select_dtypes(include=["float64"]).columns:
        result[column] = result[column].astype("float32")

    memory_mb = result.memory_usage(deep=True).sum() / 1024**2
    report.log(
        f"Optimización de tipos (categóricos y punto flotante de 32 bits): el dataset "
        f"analítico ocupa {memory_mb:.1f} MB en memoria."
    )
    return result


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

#: Orden final de columnas: identificación temporal, territorial, variables
#: observadas, variables derivadas y, al final, control de calidad y validación.
FINAL_COLUMN_ORDER: list[str] = [
    # Temporal
    "fecha", "anio", "mes", "mes_nombre", "mes_abrev", "trimestre", "dia_anio",
    "semana_anio", "anio_mes", "temporada", "campania_agricola",
    # Territorial
    "provincia", "capital", "latitud", "longitud", "altitud",
    "cuenca", "zona_agroecologica", "piso_ecologico",
    # Térmicas
    "temperatura_minima", "temperatura_media", "temperatura_maxima", "oscilacion_termica",
    "temperatura_suelo",
    # Humedad y precipitación
    "punto_rocio", "humedad_relativa", "deficit_presion_vapor", "deficit_presion_vapor_maximo",
    "precipitacion", "lluvia", "nevada", "horas_precipitacion", "humedad_suelo",
    "evapotranspiracion",
    # Radiación y viento
    "radiacion_solar", "horas_sol", "nubosidad", "velocidad_viento", "racha_viento_maxima",
    "presion_superficie",
    # Indicadores de helada
    "helada", "helada_agronomica", "intensidad_helada", "deficit_termico",
    "indice_riesgo_helada", "nivel_riesgo", "dia_lluvia", "dia_nevada",
    # Control de calidad y validación cruzada
    "precipitacion_sospechosa", "temperatura_minima_merra2", "helada_merra2", "fuente",
]


def clean_dataset(
    raw: pd.DataFrame, validation: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Ejecuta el pipeline completo de limpieza, integración y transformación.

    Args:
        raw: Dataset primario tal como se descargó del servicio.
        validation: Dataset secundario crudo para la validación cruzada. Si es
            ``None``, la etapa de integración se omite y se registra en la bitácora.

    Returns:
        Tupla ``(dataset_limpio, bitácora)``.
    """
    report = CleaningReport(rows_input=len(raw), columns_input=len(raw.columns))
    report.log(f"Inicio del pipeline: {len(raw):,} filas × {len(raw.columns)} columnas de entrada.")

    frame = _normalize_schema(raw, report)
    frame = _parse_dates(frame, report)
    frame = _harmonize_units(frame, report)
    frame = _replace_sentinels(frame, report)
    frame = _enforce_physical_ranges(frame, report)
    frame = _enforce_temperature_consistency(frame, report)
    frame = _flag_precipitation_outliers(frame, report)
    frame = _drop_duplicates(frame, report)
    frame = _reindex_calendar(frame, report)
    frame = _impute_missing(frame, report)
    frame = _merge_validation_source(frame, validation, report)
    frame = add_derived_variables(frame, report)

    frame = frame.loc[:, [column for column in FINAL_COLUMN_ORDER if column in frame.columns]]
    frame = frame.sort_values(["fecha", "provincia"]).reset_index(drop=True)
    frame = _optimize_dtypes(frame, report)

    report.rows_output = len(frame)
    report.columns_output = len(frame.columns)
    report.log(
        f"Pipeline finalizado: {report.rows_output:,} filas × {report.columns_output} columnas "
        "en el dataset analítico."
    )
    return frame, report


def build_data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    """Construye el diccionario de datos con estadísticos descriptivos reales.

    Args:
        frame: Dataset limpio.

    Returns:
        Dataframe con una fila por variable: etiqueta, unidad, tipo lógico,
        procedencia, definición, completitud y estadísticos básicos.
    """
    documented = {spec.name: spec for spec in VARIABLE_DICTIONARY}
    rows: list[dict[str, Any]] = []

    for name in frame.columns:
        spec = documented.get(name)
        series = frame[name]
        row: dict[str, Any] = {
            "variable": name,
            "etiqueta": spec.label if spec else name.replace("_", " ").capitalize(),
            "unidad": (spec.unit or "—") if spec else "—",
            "tipo": spec.dtype if spec else _infer_logical_type(series),
            "procedencia": spec.origin if spec else "derivada",
            "descripcion": spec.description if spec else "Variable auxiliar del modelo de datos.",
            "tipo_pandas": str(series.dtype),
            "completitud_%": round(100.0 * series.notna().mean(), 2),
            "valores_unicos": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            row |= {
                "minimo": round(float(series.min()), 3),
                "maximo": round(float(series.max()), 3),
                "media": round(float(series.mean()), 3),
                "mediana": round(float(series.median()), 3),
                "desv_estandar": round(float(series.std()), 3),
            }
        else:
            row |= dict.fromkeys(("minimo", "maximo", "media", "mediana", "desv_estandar"), "—")
        rows.append(row)

    return pd.DataFrame(rows)


def _infer_logical_type(series: pd.Series) -> str:
    """Deduce el tipo lógico de una serie no documentada explícitamente.

    Args:
        series: Serie a clasificar.

    Returns:
        Uno de ``booleana``, ``temporal``, ``numerica`` o ``categorica``.
    """
    if pd.api.types.is_bool_dtype(series):
        return "booleana"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"
    if pd.api.types.is_numeric_dtype(series):
        return "numerica"
    return "categorica"
