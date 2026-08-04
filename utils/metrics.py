"""Cálculo de indicadores, agregaciones y rankings del dashboard.

Todas las métricas se calculan **sobre el subconjunto filtrado**, nunca sobre el
dataset completo: es el requisito de que los indicadores se actualicen
automáticamente al modificar los filtros. Las funciones son puras y no dependen
de Streamlit, lo que permite verificarlas desde el cuaderno y desde las pruebas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from config.settings import FROST_THRESHOLDS, MONTH_ABBREV
from utils.formatting import (
    format_integer,
    format_number,
    format_percent,
    format_temperature,
    variable_unit,
)


@dataclass(frozen=True)
class Kpi:
    """Indicador clave listo para presentarse en una tarjeta.

    Attributes:
        key: Identificador estable del indicador.
        label: Título mostrado al usuario.
        value: Valor principal ya formateado.
        caption: Contexto o detalle secundario.
        delta: Variación respecto al periodo de comparación, ya formateada.
        delta_direction: ``"up"``, ``"down"`` o ``"flat"``.
        delta_is_adverse: ``True`` si la variación es desfavorable. Se separa del
            sentido porque en este dominio «más heladas» sube y es malo, mientras
            que «más temperatura mínima» sube y es bueno: el color no puede
            derivarse del signo.
        icon: Emoji identificativo del indicador.
        help_text: Definición operativa, mostrada en la ayuda contextual.
    """

    key: str
    label: str
    value: str
    caption: str = ""
    delta: str | None = None
    delta_direction: Literal["up", "down", "flat"] = "flat"
    delta_is_adverse: bool = False
    icon: str = ""
    help_text: str = ""


def _period_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el subconjunto en dos mitades temporales de igual duración.

    La comparación entre mitades es la única forma de calcular una «variación
    entre periodos» que siga siendo válida cualquiera que sea el rango de fechas
    que el usuario haya seleccionado.

    Args:
        frame: Subconjunto filtrado, con la columna ``fecha``.

    Returns:
        Tupla ``(primera_mitad, segunda_mitad)``. Ambas pueden estar vacías si el
        rango es demasiado corto.
    """
    if frame.empty:
        return frame, frame
    start, end = frame["fecha"].min(), frame["fecha"].max()
    midpoint = start + (end - start) / 2
    return frame.loc[frame["fecha"] <= midpoint], frame.loc[frame["fecha"] > midpoint]


def _relative_change(current: float, previous: float) -> float | None:
    """Calcula la variación relativa entre dos valores.

    Args:
        current: Valor del periodo reciente.
        previous: Valor del periodo anterior.

    Returns:
        Variación en porcentaje, o ``None`` si la base es nula o no válida.
    """
    if previous is None or pd.isna(previous) or previous == 0:
        return None
    return 100.0 * (current - previous) / abs(previous)


def compute_kpis(frame: pd.DataFrame, *, frost_column: str = "helada") -> list[Kpi]:
    """Calcula la batería de indicadores principales del dashboard.

    Args:
        frame: Subconjunto filtrado.
        frost_column: Columna booleana de helada correspondiente al umbral activo.

    Returns:
        Lista de indicadores. Si el subconjunto está vacío se devuelve una lista
        vacía, y es la capa de presentación la que muestra el estado sin datos.
    """
    if frame.empty:
        return []

    threshold = FROST_THRESHOLDS["meteorologica" if frost_column == "helada" else "agronomica"]
    first_half, second_half = _period_halves(frame)

    total_records = len(frame)
    frost_days = int(frame[frost_column].sum())
    frost_rate = 100.0 * frame[frost_column].mean()
    mean_minimum = float(frame["temperatura_minima"].mean())
    absolute_minimum = float(frame["temperatura_minima"].min())

    coldest = frame.loc[frame["temperatura_minima"].idxmin()]
    n_provinces = int(frame["provincia"].nunique())
    n_days = int(frame["fecha"].nunique())

    # Variación de la tasa de heladas entre la primera y la segunda mitad.
    frost_delta: float | None = None
    if not first_half.empty and not second_half.empty:
        frost_delta = _relative_change(
            100.0 * second_half[frost_column].mean(), 100.0 * first_half[frost_column].mean()
        )

    minimum_delta: float | None = None
    if not first_half.empty and not second_half.empty:
        minimum_delta = float(
            second_half["temperatura_minima"].mean() - first_half["temperatura_minima"].mean()
        )

    # Provincia con mayor incidencia relativa de heladas.
    by_province = (
        frame.groupby("provincia", observed=True)[frost_column]
        .mean()
        .mul(100.0)
        .sort_values(ascending=False)
    )
    worst_province = by_province.index[0] if len(by_province) else "—"
    worst_province_rate = float(by_province.iloc[0]) if len(by_province) else float("nan")

    annual_precipitation = (
        frame.groupby(["provincia", "anio"], observed=True)["precipitacion"].sum().groupby("anio").mean().mean()
        if "precipitacion" in frame.columns
        else float("nan")
    )

    kpis: list[Kpi] = [
        Kpi(
            key="registros",
            label="Registros analizados",
            value=format_integer(total_records),
            caption=f"{format_integer(n_days)} días × {n_provinces} provincia(s)",
            icon="🗂️",
            help_text=(
                "Número de observaciones diarias por provincia que quedan tras aplicar "
                "los filtros activos. Es el denominador de todos los demás indicadores."
            ),
        ),
        Kpi(
            key="dias_helada",
            label="Días con helada",
            value=format_integer(frost_days),
            caption=f"{format_percent(frost_rate)} del total analizado",
            delta=format_percent(frost_delta, signed=True) if frost_delta is not None else None,
            delta_direction=_direction(frost_delta),
            # Más heladas es siempre desfavorable, con independencia del signo.
            delta_is_adverse=bool(frost_delta is not None and frost_delta > 0),
            icon="❄️",
            help_text=(
                f"Días en que la temperatura mínima descendió a {threshold:.0f} °C o menos. "
                "La variación compara la segunda mitad del periodo seleccionado con la primera."
            ),
        ),
        Kpi(
            key="tmin_promedio",
            label="Temperatura mínima media",
            value=format_temperature(mean_minimum),
            caption=f"mediana {format_temperature(float(frame['temperatura_minima'].median()))}",
            delta=format_temperature(minimum_delta, signed=True) if minimum_delta is not None else None,
            delta_direction=_direction(minimum_delta),
            # Una mínima que baja agrava el riesgo de helada.
            delta_is_adverse=bool(minimum_delta is not None and minimum_delta < 0),
            icon="🌡️",
            help_text=(
                "Promedio de la temperatura mínima diaria del aire a 2 m sobre el "
                "subconjunto seleccionado."
            ),
        ),
        Kpi(
            key="tmin_absoluta",
            label="Mínima absoluta registrada",
            value=format_temperature(absolute_minimum),
            caption=f"{coldest['provincia']} · {coldest['fecha']:%d/%m/%Y}",
            icon="🥶",
            help_text=(
                "Valor más bajo de temperatura mínima en el subconjunto, con la provincia "
                "y la fecha en que se registró."
            ),
        ),
        Kpi(
            key="provincia_critica",
            label="Provincia más expuesta",
            value=str(worst_province),
            caption=f"{format_percent(worst_province_rate)} de días con helada",
            icon="📍",
            help_text=(
                "Provincia con la mayor proporción de días con helada. Se usa la proporción "
                "y no el conteo absoluto para que la comparación sea justa cuando los "
                "filtros dejan distinto número de días por provincia."
            ),
        ),
        Kpi(
            key="oscilacion",
            label="Oscilación térmica media",
            value=format_temperature(float(frame["oscilacion_termica"].mean())),
            caption=f"máxima {format_temperature(float(frame['oscilacion_termica'].max()))}",
            icon="📊",
            help_text=(
                "Diferencia media entre la temperatura máxima y la mínima del día. Una "
                "oscilación amplia delata cielos despejados, condición necesaria de la "
                "helada por radiación."
            ),
        ),
        Kpi(
            key="riesgo",
            label="Índice de riesgo medio",
            value=format_number(float(frame["indice_riesgo_helada"].mean()), decimals=1),
            caption=f"escala 0–100 · p95 = {format_number(float(frame['indice_riesgo_helada'].quantile(0.95)), decimals=1)}",
            icon="⚠️",
            help_text=(
                "Índice compuesto que integra déficit térmico, cielo despejado, sequedad "
                "atmosférica y calma de viento. Sus límites de normalización son fijos, por "
                "lo que los valores son comparables entre selecciones distintas."
            ),
        ),
        Kpi(
            key="precipitacion",
            label="Precipitación anual media",
            value=format_number(float(annual_precipitation), decimals=0, unit="mm"),
            caption="promedio por provincia y año",
            icon="🌧️",
            help_text=(
                "Acumulado anual medio de precipitación por provincia. Contextualiza el "
                "régimen hídrico en el que se producen las heladas."
            ),
        ),
    ]
    return kpis


def _direction(delta: float | None) -> Literal["up", "down", "flat"]:
    """Traduce una variación numérica en un sentido cualitativo.

    Args:
        delta: Variación observada.

    Returns:
        ``"up"``, ``"down"`` o ``"flat"``. Las variaciones muy pequeñas se
        consideran planas para no dibujar flechas sobre ruido.
    """
    if delta is None or pd.isna(delta) or abs(delta) < 0.05:
        return "flat"
    return "up" if delta > 0 else "down"


# ---------------------------------------------------------------------------
# Agregaciones
# ---------------------------------------------------------------------------


def aggregate_by_province(frame: pd.DataFrame, *, frost_column: str = "helada") -> pd.DataFrame:
    """Resume el comportamiento agroclimático de cada provincia.

    Args:
        frame: Subconjunto filtrado.
        frost_column: Columna booleana de helada del umbral activo.

    Returns:
        Dataframe con una fila por provincia, ordenado por días de helada
        descendente, incluyendo la posición en el ranking.
    """
    if frame.empty:
        return pd.DataFrame()

    summary = (
        frame.groupby("provincia", observed=True)
        .agg(
            dias_analizados=("fecha", "nunique"),
            dias_helada=(frost_column, "sum"),
            tasa_helada=(frost_column, "mean"),
            tmin_media=("temperatura_minima", "mean"),
            tmin_absoluta=("temperatura_minima", "min"),
            tmax_media=("temperatura_maxima", "mean"),
            oscilacion_media=("oscilacion_termica", "mean"),
            precipitacion_total=("precipitacion", "sum"),
            humedad_media=("humedad_relativa", "mean"),
            nubosidad_media=("nubosidad", "mean"),
            radiacion_media=("radiacion_solar", "mean"),
            viento_medio=("velocidad_viento", "mean"),
            riesgo_medio=("indice_riesgo_helada", "mean"),
            deficit_termico_total=("deficit_termico", "sum"),
            altitud=("altitud", "first"),
            latitud=("latitud", "first"),
            longitud=("longitud", "first"),
            capital=("capital", "first"),
            cuenca=("cuenca", "first"),
            piso_ecologico=("piso_ecologico", "first"),
        )
        .reset_index()
    )

    summary["tasa_helada"] = (summary["tasa_helada"] * 100.0).round(2)
    n_years = max(1.0, frame["fecha"].nunique() / 365.25)
    summary["heladas_por_anio"] = (summary["dias_helada"] / n_years).round(1)
    summary["precipitacion_anual"] = (summary["precipitacion_total"] / n_years).round(0)

    summary = summary.sort_values("dias_helada", ascending=False).reset_index(drop=True)
    summary.insert(0, "ranking", range(1, len(summary) + 1))
    return summary.round(2)


def aggregate_monthly(frame: pd.DataFrame, *, frost_column: str = "helada") -> pd.DataFrame:
    """Construye el perfil del ciclo anual promedio.

    Args:
        frame: Subconjunto filtrado.
        frost_column: Columna booleana de helada del umbral activo.

    Returns:
        Dataframe con una fila por mes calendario, en orden cronológico.
    """
    if frame.empty:
        return pd.DataFrame()

    monthly = (
        frame.groupby("mes", observed=True)
        .agg(
            dias_analizados=("fecha", "count"),
            dias_helada=(frost_column, "sum"),
            tasa_helada=(frost_column, "mean"),
            tmin_media=("temperatura_minima", "mean"),
            tmin_p05=("temperatura_minima", lambda s: s.quantile(0.05)),
            tmax_media=("temperatura_maxima", "mean"),
            oscilacion_media=("oscilacion_termica", "mean"),
            precipitacion_media=("precipitacion", "mean"),
            humedad_media=("humedad_relativa", "mean"),
            nubosidad_media=("nubosidad", "mean"),
            riesgo_medio=("indice_riesgo_helada", "mean"),
        )
        .reset_index()
    )
    monthly["tasa_helada"] = (monthly["tasa_helada"] * 100.0).round(2)
    monthly["mes_abrev"] = monthly["mes"].map(MONTH_ABBREV)
    return monthly.sort_values("mes").round(2)


def aggregate_annual(frame: pd.DataFrame, *, frost_column: str = "helada") -> pd.DataFrame:
    """Construye la serie anual empleada en el análisis de tendencia.

    El conteo de heladas se **normaliza por provincia** dividiendo entre el
    número de provincias presentes. Sin esa normalización, un año en que el
    filtro dejara menos provincias mostraría una caída espuria que se
    interpretaría como una mejora del clima.

    Args:
        frame: Subconjunto filtrado.
        frost_column: Columna booleana de helada del umbral activo.

    Returns:
        Dataframe con una fila por año, en orden cronológico.
    """
    if frame.empty:
        return pd.DataFrame()

    annual = (
        frame.groupby("anio", observed=True)
        .agg(
            dias_analizados=("fecha", "count"),
            n_provincias=("provincia", "nunique"),
            dias_helada=(frost_column, "sum"),
            tasa_helada=(frost_column, "mean"),
            tmin_media=("temperatura_minima", "mean"),
            tmin_absoluta=("temperatura_minima", "min"),
            tmax_media=("temperatura_maxima", "mean"),
            oscilacion_media=("oscilacion_termica", "mean"),
            precipitacion_total=("precipitacion", "sum"),
            riesgo_medio=("indice_riesgo_helada", "mean"),
            deficit_termico_total=("deficit_termico", "sum"),
        )
        .reset_index()
    )
    annual["tasa_helada"] = (annual["tasa_helada"] * 100.0).round(2)
    annual["heladas_por_provincia"] = (annual["dias_helada"] / annual["n_provincias"]).round(1)
    annual["precipitacion_por_provincia"] = (
        annual["precipitacion_total"] / annual["n_provincias"]
    ).round(0)
    return annual.sort_values("anio").round(2)


def frost_calendar(frame: pd.DataFrame, *, frost_column: str = "helada") -> pd.DataFrame:
    """Construye la matriz provincia × mes de incidencia de heladas.

    Args:
        frame: Subconjunto filtrado.
        frost_column: Columna booleana de helada del umbral activo.

    Returns:
        Tabla cruzada con la tasa de heladas en porcentaje. Las provincias se
        ordenan por incidencia total, de modo que el patrón resulte legible en
        lugar de quedar disperso por el orden alfabético.
    """
    if frame.empty:
        return pd.DataFrame()

    matrix = (
        frame.pivot_table(index="provincia", columns="mes", values=frost_column, aggfunc="mean", observed=True)
        .mul(100.0)
        .round(1)
    )
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    matrix.columns = [MONTH_ABBREV[int(month)] for month in matrix.columns]
    return matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]


def aggregate_time_series(
    frame: pd.DataFrame,
    *,
    value_column: str,
    frequency: Literal["D", "W", "MS", "YS"] = "MS",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Agrega una variable a la frecuencia temporal solicitada.

    Args:
        frame: Subconjunto filtrado.
        value_column: Variable a agregar.
        frequency: Frecuencia de remuestreo de pandas.
        aggregation: Función de agregación (``mean``, ``sum``, ``min``, ``max``).

    Returns:
        Dataframe con las columnas ``fecha`` y ``valor``.
    """
    if frame.empty or value_column not in frame.columns:
        return pd.DataFrame(columns=["fecha", "valor"])

    series = (
        frame.set_index("fecha")[value_column].resample(frequency).agg(aggregation).dropna().reset_index()
    )
    return series.rename(columns={value_column: "valor"})


def rank_provinces(
    frame: pd.DataFrame, *, metric: str, ascending: bool = False, limit: int | None = None
) -> pd.DataFrame:
    """Ordena las provincias según una métrica agregada.

    Args:
        frame: Resumen provincial producido por :func:`aggregate_by_province`.
        metric: Columna por la que se ordena.
        ascending: Sentido del orden.
        limit: Número máximo de filas a devolver.

    Returns:
        Dataframe ordenado con la posición recalculada.
    """
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame()

    ranked = frame.sort_values(metric, ascending=ascending).reset_index(drop=True)
    ranked["posicion"] = range(1, len(ranked) + 1)
    return ranked.head(limit) if limit else ranked


def compare_sources(frame: pd.DataFrame) -> dict[str, float] | None:
    """Cuantifica la concordancia entre la fuente primaria y la de validación.

    Args:
        frame: Subconjunto filtrado que incluye ``temperatura_minima_merra2``.

    Returns:
        Diccionario con las métricas de concordancia, o ``None`` si la fuente de
        validación no está disponible en el subconjunto.
    """
    required = {"temperatura_minima", "temperatura_minima_merra2"}
    if not required.issubset(frame.columns):
        return None

    paired = frame.loc[:, ["temperatura_minima", "temperatura_minima_merra2", "helada"]].dropna()
    if len(paired) < 30:
        return None

    difference = paired["temperatura_minima"] - paired["temperatura_minima_merra2"]
    secondary_frost = paired["temperatura_minima_merra2"] <= FROST_THRESHOLDS["meteorologica"]

    return {
        "n_pares": int(len(paired)),
        "correlacion": round(float(paired["temperatura_minima"].corr(paired["temperatura_minima_merra2"])), 4),
        "sesgo_medio": round(float(difference.mean()), 3),
        "error_absoluto_medio": round(float(difference.abs().mean()), 3),
        "raiz_error_cuadratico_medio": round(float(np.sqrt((difference**2).mean())), 3),
        "concordancia_helada": round(float((paired["helada"] == secondary_frost).mean() * 100.0), 2),
    }


def summary_table(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Construye la tabla de estadísticos descriptivos de varias variables.

    Args:
        frame: Subconjunto filtrado.
        variables: Variables numéricas a describir.

    Returns:
        Dataframe con una fila por variable y los estadísticos como columnas,
        listo para mostrarse como vista de tabla o exportarse.
    """
    from utils.stats_tools import describe_distribution

    rows: list[dict[str, object]] = []
    for variable in variables:
        if variable not in frame.columns:
            continue
        statistics = describe_distribution(frame[variable])
        if not statistics:
            continue
        rows.append(
            {
                "Variable": variable,
                "Unidad": variable_unit(variable) or "—",
                "n": int(statistics["n"]),
                "Media": round(statistics["media"], 2),
                "Mediana": round(statistics["mediana"], 2),
                "Desv. típica": round(statistics["desv_estandar"], 2),
                "Mínimo": round(statistics["minimo"], 2),
                "P5": round(statistics["p05"], 2),
                "Q1": round(statistics["q1"], 2),
                "Q3": round(statistics["q3"], 2),
                "P95": round(statistics["p95"], 2),
                "Máximo": round(statistics["maximo"], 2),
                "Asimetría": round(statistics["asimetria"], 2),
                "Curtosis": round(statistics["curtosis"], 2),
            }
        )
    return pd.DataFrame(rows)
