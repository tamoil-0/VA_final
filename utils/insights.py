"""Motor de hallazgos: convierte resultados numéricos en conclusiones redactadas.

Este módulo materializa un requisito explícito de la consigna: cada sección del
dashboard debe indicar qué representa la visualización, qué patrón se observa,
qué hallazgo es relevante, qué decisión puede derivarse y qué limitación
presentan los datos. En lugar de escribir esos textos a mano —lo que los volvería
falsos en cuanto el usuario mueva un filtro— se generan a partir del subconjunto
activo, de modo que la interpretación siempre corresponde a lo que hay en
pantalla.

Cada hallazgo se modela como un :class:`Insight`, con severidad y evidencia
numérica asociada, para que la capa de presentación decida su tratamiento visual
sin volver a razonar sobre los datos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from config.settings import FROST_THRESHOLDS, MONTH_LABELS
from utils.formatting import (
    format_integer,
    format_number,
    format_percent,
    format_pvalue,
    format_temperature,
    variable_label,
)
from utils.metrics import aggregate_annual, aggregate_by_province, aggregate_monthly
from utils.stats_tools import (
    ALPHA,
    fit_linear_trend,
    mann_kendall_test,
    top_correlations,
)

Severity = Literal["info", "good", "warning", "serious", "critical"]


@dataclass(frozen=True)
class Insight:
    """Hallazgo detectado automáticamente sobre el subconjunto activo.

    Attributes:
        title: Enunciado corto del hallazgo.
        body: Explicación con la evidencia numérica que lo sostiene.
        severity: Nivel de atención que merece.
        icon: Emoji identificativo.
        evidence: Cifras que respaldan el hallazgo, para trazabilidad.
        recommendation: Decisión o acción que se deriva del hallazgo.
    """

    title: str
    body: str
    severity: Severity = "info"
    icon: str = "💡"
    evidence: dict[str, float | str] | None = None
    recommendation: str = ""


@dataclass(frozen=True)
class SectionInterpretation:
    """Interpretación completa de una sección del dashboard.

    Cubre los cinco puntos que la consigna exige para cada visualización.

    Attributes:
        represents: Qué representa la visualización.
        pattern: Qué patrón o comportamiento se observa.
        finding: Qué hallazgo es relevante.
        decision: Qué decisión o recomendación puede derivarse.
        limitation: Qué limitación presentan los datos.
    """

    represents: str
    pattern: str
    finding: str
    decision: str
    limitation: str


# ---------------------------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------------------------


def build_executive_summary(frame: pd.DataFrame, *, frost_column: str = "helada") -> str:
    """Redacta el resumen ejecutivo del subconjunto activo.

    Args:
        frame: Subconjunto filtrado.
        frost_column: Columna booleana de helada del umbral activo.

    Returns:
        Párrafo en prosa que sintetiza la situación general del fenómeno.
    """
    if frame.empty:
        return "La selección actual no contiene registros. Amplíe el rango de filtros para obtener resultados."

    threshold = FROST_THRESHOLDS["meteorologica" if frost_column == "helada" else "agronomica"]
    provinces = aggregate_by_province(frame, frost_column=frost_column)
    monthly = aggregate_monthly(frame, frost_column=frost_column)

    frost_rate = 100.0 * frame[frost_column].mean()
    mean_minimum = float(frame["temperatura_minima"].mean())
    start, end = frame["fecha"].min(), frame["fecha"].max()

    peak_month = int(monthly.loc[monthly["tasa_helada"].idxmax(), "mes"]) if not monthly.empty else 0
    calm_month = int(monthly.loc[monthly["tasa_helada"].idxmin(), "mes"]) if not monthly.empty else 0

    most_exposed = provinces.iloc[0] if not provinces.empty else None
    least_exposed = provinces.iloc[-1] if not provinces.empty else None

    text = (
        f"Entre el {start:%d/%m/%Y} y el {end:%d/%m/%Y} se analizaron "
        f"**{format_integer(len(frame))} observaciones diarias** correspondientes a "
        f"**{frame['provincia'].nunique()} provincia(s)** de la región Puno. En ese periodo, "
        f"**{format_percent(frost_rate)}** de los días registró helada (temperatura mínima ≤ "
        f"{threshold:.0f} °C), con una temperatura mínima promedio de "
        f"**{format_temperature(mean_minimum)}**."
    )

    if peak_month and calm_month:
        text += (
            f" El fenómeno presenta una estacionalidad marcada: **{MONTH_LABELS[peak_month]}** "
            f"concentra la mayor incidencia ({format_percent(float(monthly['tasa_helada'].max()))} "
            f"de los días) y **{MONTH_LABELS[calm_month]}** la menor "
            f"({format_percent(float(monthly['tasa_helada'].min()))})."
        )

    if most_exposed is not None and least_exposed is not None and len(provinces) > 1:
        text += (
            f" Territorialmente, **{most_exposed['provincia']}** "
            f"({format_number(most_exposed['altitud'], decimals=0, unit='msnm')}) es la más expuesta, "
            f"con {format_percent(most_exposed['tasa_helada'])} de días con helada, frente a "
            f"**{least_exposed['provincia']}** "
            f"({format_number(least_exposed['altitud'], decimals=0, unit='msnm')}), con "
            f"{format_percent(least_exposed['tasa_helada'])}. La brecha entre ambos extremos es de "
            f"**{format_number(most_exposed['tasa_helada'] - least_exposed['tasa_helada'], decimals=1, unit='puntos porcentuales')}**."
        )

    annual = aggregate_annual(frame, frost_column=frost_column)
    if len(annual) >= 4:
        test = mann_kendall_test(annual["heladas_por_provincia"])
        if test is not None and test.is_significant:
            direction = "al alza" if test.trend == "creciente" else "a la baja"
            text += (
                f" La serie anual muestra una tendencia **{direction} estadísticamente "
                f"significativa** ({format_pvalue(test.p_value)}), con una pendiente de Sen de "
                f"{format_number(test.sen_slope, decimals=2, signed=True)} días de helada por año."
            )
        elif test is not None:
            text += (
                " La serie anual **no presenta una tendencia estadísticamente significativa** "
                f"en el periodo analizado ({format_pvalue(test.p_value)}), lo que sugiere un "
                "régimen estable antes que un cambio direccional."
            )

    return text


# ---------------------------------------------------------------------------
# Hallazgos automáticos
# ---------------------------------------------------------------------------


def detect_insights(
    frame: pd.DataFrame, *, frost_column: str = "helada", limit: int = 8
) -> list[Insight]:
    """Genera la lista de hallazgos relevantes del subconjunto activo.

    Args:
        frame: Subconjunto filtrado.
        frost_column: Columna booleana de helada del umbral activo.
        limit: Número máximo de hallazgos a devolver.

    Returns:
        Lista de hallazgos ordenada por severidad descendente.
    """
    if frame.empty:
        return []

    insights: list[Insight] = []
    provinces = aggregate_by_province(frame, frost_column=frost_column)
    monthly = aggregate_monthly(frame, frost_column=frost_column)
    annual = aggregate_annual(frame, frost_column=frost_column)

    insights.extend(_insight_territorial_gap(provinces))
    insights.extend(_insight_altitude_gradient(provinces))
    insights.extend(_insight_seasonality(monthly))
    insights.extend(_insight_trend(annual))
    insights.extend(_insight_extreme_event(frame))
    insights.extend(_insight_severity_mix(frame))
    insights.extend(_insight_drivers(frame))
    insights.extend(_insight_data_quality(frame))
    insights.extend(_insight_source_agreement(frame))

    order: dict[Severity, int] = {"critical": 0, "serious": 1, "warning": 2, "good": 3, "info": 4}
    insights.sort(key=lambda insight: order.get(insight.severity, 5))
    return insights[:limit]


def _insight_territorial_gap(provinces: pd.DataFrame) -> list[Insight]:
    """Detecta la desigualdad territorial en la exposición a heladas."""
    if len(provinces) < 2:
        return []

    top = provinces.iloc[0]
    bottom = provinces.iloc[-1]
    gap = float(top["tasa_helada"] - bottom["tasa_helada"])
    if gap < 5:
        return []

    ratio = float(top["tasa_helada"] / bottom["tasa_helada"]) if bottom["tasa_helada"] > 0 else float("inf")
    ratio_text = (
        f"{format_number(ratio, decimals=1)} veces"
        if np.isfinite(ratio)
        else "un orden de magnitud imposible de expresar como razón, al no registrarse heladas en la provincia menos expuesta"
    )

    return [
        Insight(
            title="La exposición al riesgo es profundamente desigual entre provincias",
            body=(
                f"**{top['provincia']}** registra helada en {format_percent(top['tasa_helada'])} de los días "
                f"analizados, mientras que **{bottom['provincia']}** lo hace en "
                f"{format_percent(bottom['tasa_helada'])}: una diferencia de "
                f"{format_number(gap, decimals=1, unit='puntos porcentuales')} y una razón de {ratio_text}. "
                "La región no constituye, por tanto, una unidad homogénea de riesgo."
            ),
            severity="serious" if gap > 30 else "warning",
            icon="🗺️",
            evidence={
                "provincia_maxima": str(top["provincia"]),
                "tasa_maxima": float(top["tasa_helada"]),
                "provincia_minima": str(bottom["provincia"]),
                "tasa_minima": float(bottom["tasa_helada"]),
                "brecha_pp": round(gap, 2),
            },
            recommendation=(
                "Diferenciar territorialmente los programas de mitigación en lugar de aplicar "
                "una política regional uniforme: concentrar el seguro agrario, los sistemas de "
                "alerta y la asistencia técnica en las provincias del extremo superior."
            ),
        )
    ]


def _insight_altitude_gradient(provinces: pd.DataFrame) -> list[Insight]:
    """Cuantifica la relación entre altitud y exposición a heladas."""
    if len(provinces) < 5 or "altitud" not in provinces.columns:
        return []

    trend = fit_linear_trend(provinces["altitud"], provinces["tmin_media"])
    if trend is None:
        return []

    # El gradiente térmico vertical se expresa convencionalmente por cada 100 m.
    gradient_per_100m = trend.slope * 100.0

    if not trend.is_significant:
        return [
            Insight(
                title="La altitud no explica por sí sola la exposición al riesgo",
                body=(
                    "La relación entre la altitud de la capital provincial y su temperatura mínima "
                    f"media no alcanza significancia estadística ({format_pvalue(trend.p_value)}, "
                    f"R² = {format_number(trend.r_squared, decimals=2, thousands=False)}). Con las "
                    "provincias seleccionadas, factores locales —la influencia térmica del lago "
                    "Titicaca, la orientación del valle o la exposición al viento— pesan más que "
                    "la elevación."
                ),
                severity="info",
                icon="⛰️",
                evidence={"r2": round(trend.r_squared, 3), "p": round(trend.p_value, 4)},
                recommendation=(
                    "No usar la altitud como único criterio de zonificación del riesgo; "
                    "incorporar la distancia al lago y la posición fisiográfica."
                ),
            )
        ]

    return [
        Insight(
            title="Existe un gradiente altitudinal claro del riesgo térmico",
            body=(
                f"Por cada 100 m de ascenso, la temperatura mínima media descansa "
                f"{format_temperature(abs(gradient_per_100m), decimals=2)} "
                f"{'menos' if gradient_per_100m < 0 else 'más'}. La relación explica el "
                f"{format_percent(trend.r_squared * 100)} de la variabilidad entre provincias y es "
                f"estadísticamente significativa ({format_pvalue(trend.p_value)}). El resultado es "
                "coherente con el gradiente adiabático esperado en los Andes tropicales."
            ),
            severity="info",
            icon="⛰️",
            evidence={
                "gradiente_c_por_100m": round(gradient_per_100m, 3),
                "r2": round(trend.r_squared, 3),
                "p": round(trend.p_value, 6),
            },
            recommendation=(
                "Emplear la cota altitudinal como primer criterio de zonificación del riesgo "
                "agroclimático y de selección de variedades resistentes al frío."
            ),
        )
    ]


def _insight_seasonality(monthly: pd.DataFrame) -> list[Insight]:
    """Describe la concentración estacional del fenómeno."""
    if monthly.empty or len(monthly) < 6:
        return []

    ordered = monthly.sort_values("tasa_helada", ascending=False)
    critical = ordered.head(3)
    concentration = float(critical["dias_helada"].sum() / max(1.0, monthly["dias_helada"].sum()) * 100.0)
    month_names = [MONTH_LABELS[int(month)] for month in critical["mes"]]

    return [
        Insight(
            title="El riesgo se concentra en una ventana estacional estrecha",
            body=(
                f"Los meses de **{', '.join(month_names)}** acumulan el "
                f"{format_percent(concentration)} de todos los días con helada del periodo, pese a "
                "representar sólo la cuarta parte del año. Esa concentración coincide con la "
                "estación seca del altiplano, cuando el cielo despejado maximiza la pérdida "
                "radiativa nocturna."
            ),
            severity="warning" if concentration > 55 else "info",
            icon="📅",
            evidence={
                "meses_criticos": ", ".join(month_names),
                "concentracion_%": round(concentration, 2),
            },
            recommendation=(
                "Concentrar en esa ventana los recursos operativos: activación de alertas "
                "tempranas, disponibilidad de insumos de protección y calendarización de "
                "siembras que evite la coincidencia con las fases fenológicas más sensibles."
            ),
        )
    ]


def _insight_trend(annual: pd.DataFrame) -> list[Insight]:
    """Evalúa la existencia de tendencia interanual con dos métodos."""
    if len(annual) < 5:
        return []

    non_parametric = mann_kendall_test(annual["heladas_por_provincia"])
    parametric = fit_linear_trend(annual["anio"], annual["heladas_por_provincia"])
    if non_parametric is None or parametric is None:
        return []

    agree = (non_parametric.is_significant and parametric.is_significant) and (
        np.sign(non_parametric.sen_slope) == np.sign(parametric.slope)
    )

    if agree:
        rising = non_parametric.sen_slope > 0
        decade_change = non_parametric.sen_slope * 10
        return [
            Insight(
                title=(
                    "Los días de helada aumentan de forma sostenida"
                    if rising
                    else "Los días de helada disminuyen de forma sostenida"
                ),
                body=(
                    "Las dos pruebas de tendencia coinciden. Mann-Kendall detecta una tendencia "
                    f"{non_parametric.trend} ({format_pvalue(non_parametric.p_value)}, τ = "
                    f"{format_number(non_parametric.tau, decimals=2, thousands=False)}) y la "
                    f"regresión lineal la confirma ({format_pvalue(parametric.p_value)}, R² = "
                    f"{format_number(parametric.r_squared, decimals=2, thousands=False)}). "
                    f"La pendiente de Sen equivale a "
                    f"{format_number(decade_change, decimals=1, signed=True)} días de helada por "
                    "década y por provincia. Que ambos métodos —uno paramétrico y otro no— "
                    "converjan hace la conclusión robusta frente a valores atípicos."
                ),
                severity="critical" if rising else "good",
                icon="📈" if rising else "📉",
                evidence={
                    "pendiente_sen_dias_por_anio": round(non_parametric.sen_slope, 3),
                    "tau": round(non_parametric.tau, 3),
                    "p_mann_kendall": round(non_parametric.p_value, 5),
                    "r2_ols": round(parametric.r_squared, 3),
                },
                recommendation=(
                    "Incorporar la tendencia a la planificación agrícola de mediano plazo: "
                    "revisar los calendarios de siembra y priorizar la investigación en "
                    "variedades tolerantes al frío."
                    if rising
                    else "Mantener el monitoreo: una tendencia favorable no elimina el riesgo de "
                    "eventos extremos puntuales, que siguen siendo los que causan pérdidas."
                ),
            )
        ]

    if non_parametric.is_significant != parametric.is_significant:
        return [
            Insight(
                title="Las pruebas de tendencia no concuerdan entre sí",
                body=(
                    f"Mann-Kendall concluye «{non_parametric.trend}» "
                    f"({format_pvalue(non_parametric.p_value)}) mientras que la regresión lineal "
                    f"resulta {'significativa' if parametric.is_significant else 'no significativa'} "
                    f"({format_pvalue(parametric.p_value)}). La discrepancia indica que el "
                    "resultado depende del método y, por tanto, que la evidencia de tendencia es "
                    "frágil: probablemente unos pocos años extremos gobiernan el ajuste lineal."
                ),
                severity="warning",
                icon="⚖️",
                evidence={
                    "p_mann_kendall": round(non_parametric.p_value, 5),
                    "p_ols": round(parametric.p_value, 5),
                },
                recommendation=(
                    "No sustentar decisiones de largo plazo en esta serie: extender el periodo de "
                    "observación antes de afirmar la existencia de una tendencia."
                ),
            )
        ]

    return [
        Insight(
            title="El régimen de heladas se mantiene estable en el periodo",
            body=(
                "Ninguna de las dos pruebas detecta tendencia significativa "
                f"(Mann-Kendall {format_pvalue(non_parametric.p_value)}; regresión "
                f"{format_pvalue(parametric.p_value)}). La variabilidad observada entre años es "
                "compatible con fluctuación interanual y no con un cambio direccional."
            ),
            severity="info",
            icon="➖",
            evidence={
                "p_mann_kendall": round(non_parametric.p_value, 5),
                "p_ols": round(parametric.p_value, 5),
            },
            recommendation=(
                "Dimensionar las políticas de mitigación sobre la variabilidad histórica "
                "observada, que es el riesgo real, y no sobre una tendencia inexistente."
            ),
        )
    ]


def _insight_extreme_event(frame: pd.DataFrame) -> list[Insight]:
    """Identifica el evento térmico más severo del subconjunto."""
    if frame.empty or "temperatura_minima" not in frame.columns:
        return []

    coldest = frame.loc[frame["temperatura_minima"].idxmin()]
    value = float(coldest["temperatura_minima"])
    percentile = float((frame["temperatura_minima"] <= value).mean() * 100)

    return [
        Insight(
            title="Evento térmico más severo del periodo seleccionado",
            body=(
                f"El registro más frío corresponde a **{coldest['provincia']}** el "
                f"**{coldest['fecha']:%d/%m/%Y}**, con {format_temperature(value)}. Ese día la "
                f"oscilación térmica alcanzó "
                f"{format_temperature(float(coldest['oscilacion_termica']))} y la nubosidad media "
                f"fue de {format_percent(float(coldest['nubosidad']))}, combinación característica "
                "de una helada de radiación: cielo despejado y fuerte enfriamiento nocturno. "
                f"Sólo el {format_percent(percentile, decimals=2)} de las observaciones alcanza o "
                "supera esa severidad."
            ),
            severity="serious",
            icon="🥶",
            evidence={
                "provincia": str(coldest["provincia"]),
                "fecha": f"{coldest['fecha']:%Y-%m-%d}",
                "temperatura_minima": round(value, 2),
                "nubosidad": round(float(coldest["nubosidad"]), 1),
            },
            recommendation=(
                "Usar este episodio como escenario de referencia para dimensionar las medidas "
                "de protección de cultivos y para el diseño de simulacros de respuesta."
            ),
        )
    ]


def _insight_severity_mix(frame: pd.DataFrame) -> list[Insight]:
    """Analiza la composición por severidad de los eventos de helada."""
    if "intensidad_helada" not in frame.columns:
        return []

    counts = frame["intensidad_helada"].value_counts()
    frost_events = int(counts.drop(labels=["Sin helada"], errors="ignore").sum())
    if frost_events == 0:
        return [
            Insight(
                title="La selección actual no contiene eventos de helada",
                body=(
                    "Ninguna observación del subconjunto alcanza el umbral de helada. El resultado "
                    "es informativo en sí mismo: identifica las condiciones y los territorios "
                    "libres de este riesgo."
                ),
                severity="good",
                icon="✅",
                recommendation=(
                    "Considerar estas zonas y periodos como candidatos preferentes para cultivos "
                    "sensibles al frío."
                ),
            )
        ]

    severe = int(counts.get("Severa", 0) + counts.get("Extrema", 0))
    severe_share = 100.0 * severe / frost_events

    return [
        Insight(
            title="Composición por severidad de los eventos de helada",
            body=(
                f"De los {format_integer(frost_events)} eventos de helada registrados, "
                f"{format_integer(severe)} ({format_percent(severe_share)}) son de intensidad "
                "**severa o extrema** (temperatura mínima por debajo de −4 °C). Son estos, y no "
                "las heladas ligeras, los que producen daño fisiológico irreversible en los "
                "cultivos altoandinos."
            ),
            severity="critical" if severe_share > 25 else "warning" if severe_share > 10 else "info",
            icon="🌡️",
            evidence={
                "eventos_totales": frost_events,
                "eventos_severos": severe,
                "proporcion_severa_%": round(severe_share, 2),
            },
            recommendation=(
                "Dimensionar la respuesta según severidad y no según frecuencia: un territorio "
                "con pocas heladas pero muy intensas puede requerir más protección que uno con "
                "muchas heladas ligeras."
            ),
        )
    ]


def _insight_drivers(frame: pd.DataFrame) -> list[Insight]:
    """Identifica las variables más asociadas a la temperatura mínima."""
    candidates = [
        "nubosidad", "humedad_relativa", "radiacion_solar", "velocidad_viento",
        "oscilacion_termica", "punto_rocio", "altitud", "precipitacion",
        "deficit_presion_vapor", "temperatura_suelo",
    ]
    available = [column for column in candidates if column in frame.columns]
    if len(available) < 3 or len(frame) < 100:
        return []

    findings = top_correlations(
        frame, [*available, "temperatura_minima"], target="temperatura_minima", limit=3
    )
    findings = [finding for finding in findings if abs(finding.coefficient) >= 0.3]
    if not findings:
        return []

    strongest = findings[0]
    other = findings[1] if len(findings) > 1 else None
    partner = strongest.second if strongest.first == "temperatura_minima" else strongest.first

    body = (
        f"La variable más asociada a la temperatura mínima es **{variable_label(partner, with_unit=False)}**, "
        f"con una correlación {strongest.direction} {strongest.strength} "
        f"(r = {format_number(strongest.coefficient, decimals=2, thousands=False, signed=True)}, "
        f"{format_pvalue(strongest.p_value)})."
    )
    if other is not None:
        second_partner = other.second if other.first == "temperatura_minima" else other.first
        body += (
            f" Le sigue **{variable_label(second_partner, with_unit=False)}** "
            f"(r = {format_number(other.coefficient, decimals=2, thousands=False, signed=True)})."
        )
    body += (
        " La asociación no implica causalidad, pero es consistente con el mecanismo físico de la "
        "helada de radiación, en el que el cielo despejado y el aire seco favorecen la pérdida "
        "de calor nocturna."
    )

    return [
        Insight(
            title="Factores asociados a la caída de la temperatura mínima",
            body=body,
            severity="info",
            icon="🔗",
            evidence={
                "variable": partner,
                "r": round(strongest.coefficient, 3),
                "p": round(strongest.p_value, 6),
            },
            recommendation=(
                "Incorporar la nubosidad y la humedad al sistema de alerta temprana: son "
                "observables con antelación y anticipan la ocurrencia del evento."
            ),
        )
    ]


def _insight_data_quality(frame: pd.DataFrame) -> list[Insight]:
    """Advierte sobre registros marcados por el control de calidad."""
    if "precipitacion_sospechosa" not in frame.columns:
        return []

    flagged = int(frame["precipitacion_sospechosa"].sum())
    if flagged == 0:
        return []

    affected = frame.loc[frame["precipitacion_sospechosa"]]
    worst_date = affected["fecha"].value_counts().idxmax()

    return [
        Insight(
            title="El subconjunto incluye registros marcados por control de calidad",
            body=(
                f"{format_integer(flagged)} observación(es) presentan acumulados diarios de "
                "precipitación superiores a 100 mm, magnitud que las estaciones de superficie del "
                "SENAMHI no reportan en la cuenca del Titicaca. El episodio con más provincias "
                f"afectadas es el **{worst_date:%d/%m/%Y}**. Los valores se conservaron sin "
                "modificar y se marcaron explícitamente, en lugar de imputarlos, para que la "
                "limitación permanezca visible y auditable."
            ),
            severity="warning",
            icon="🔍",
            evidence={"registros_marcados": flagged, "fecha_dominante": f"{worst_date:%Y-%m-%d}"},
            recommendation=(
                "Activar el filtro «excluir registros sospechosos» al analizar el régimen "
                "hídrico. Para el análisis de heladas el efecto es nulo, ya que la marca afecta "
                "a la precipitación y no a la temperatura."
            ),
        )
    ]


def _insight_source_agreement(frame: pd.DataFrame) -> list[Insight]:
    """Reporta la concordancia con la fuente independiente de validación."""
    from utils.metrics import compare_sources

    agreement = compare_sources(frame)
    if agreement is None:
        return []

    correlation = agreement["correlacion"]
    concordance = agreement["concordancia_helada"]
    bias = agreement["sesgo_medio"]

    if correlation >= 0.85 and concordance >= 80:
        severity: Severity = "good"
        verdict = (
            "La coincidencia entre dos reanálisis producidos por instituciones distintas y con "
            "modelos distintos respalda la validez de la señal detectada."
        )
    else:
        severity = "warning"
        verdict = (
            "La concordancia moderada obliga a leer las cifras absolutas con cautela: los "
            "patrones relativos entre provincias y meses son más fiables que los valores puntuales."
        )

    return [
        Insight(
            title="Validación cruzada con una fuente independiente",
            body=(
                f"Sobre {format_integer(int(agreement['n_pares']))} pares de observaciones, la "
                f"temperatura mínima de ERA5-Land y la de MERRA-2 correlacionan a "
                f"r = {format_number(correlation, decimals=3, thousands=False)}, con un sesgo medio "
                f"de {format_temperature(bias, decimals=2, signed=True)} y un error absoluto medio "
                f"de {format_temperature(agreement['error_absoluto_medio'], decimals=2)}. "
                f"La clasificación binaria de helada coincide en el "
                f"{format_percent(concordance)} de los días. {verdict}"
            ),
            severity=severity,
            icon="🔬",
            evidence=dict(agreement),
            recommendation=(
                "Reportar siempre el intervalo entre fuentes al comunicar cifras a tomadores de "
                "decisión, en lugar de presentar un valor único como exacto."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Interpretaciones por sección
# ---------------------------------------------------------------------------


def interpret_distribution(frame: pd.DataFrame, variable: str) -> SectionInterpretation:
    """Redacta la interpretación de un histograma o diagrama de caja.

    Args:
        frame: Subconjunto filtrado.
        variable: Variable representada.

    Returns:
        Interpretación con los cinco puntos que exige la consigna.
    """
    from utils.stats_tools import describe_distribution

    statistics = describe_distribution(frame[variable]) if variable in frame.columns else {}
    label = variable_label(variable)

    if not statistics:
        return SectionInterpretation(
            represents=f"Distribución de {label}.",
            pattern="Sin datos suficientes en la selección actual.",
            finding="No es posible caracterizar la distribución.",
            decision="Amplíe el rango de filtros.",
            limitation="La selección activa no contiene observaciones válidas.",
        )

    skewness = statistics["asimetria"]
    if abs(skewness) < 0.5:
        shape = "aproximadamente simétrica"
        implication = "la media es un buen resumen del comportamiento típico"
    elif skewness > 0:
        shape = "asimétrica a la derecha, con una cola de valores altos"
        implication = "la media sobreestima el valor típico y conviene usar la mediana"
    else:
        shape = "asimétrica a la izquierda, con una cola de valores bajos"
        implication = "la media subestima el valor típico y conviene usar la mediana"

    kurtosis_note = (
        "La curtosis elevada indica más eventos extremos que en una distribución normal."
        if statistics["curtosis"] > 1
        else "La curtosis es próxima a la de una distribución normal."
    )

    return SectionInterpretation(
        represents=(
            f"Cómo se reparten las {format_integer(int(statistics['n']))} observaciones de "
            f"{label} en el subconjunto seleccionado."
        ),
        pattern=(
            f"La distribución es {shape}: media "
            f"{format_number(statistics['media'], decimals=2)} frente a mediana "
            f"{format_number(statistics['mediana'], decimals=2)}, con desviación típica "
            f"{format_number(statistics['desv_estandar'], decimals=2)}. El 90 % central se sitúa "
            f"entre {format_number(statistics['p05'], decimals=2)} y "
            f"{format_number(statistics['p95'], decimals=2)}. {kurtosis_note}"
        ),
        finding=(
            f"El recorrido total va de {format_number(statistics['minimo'], decimals=2)} a "
            f"{format_number(statistics['maximo'], decimals=2)}, y el rango interquartílico es de "
            f"{format_number(statistics['rango_interquartil'], decimals=2)}: "
            f"{implication}."
        ),
        decision=(
            "Emplear los percentiles 5 y 95 —y no el mínimo y el máximo— para fijar umbrales "
            "operativos de alerta: los extremos absolutos son eventos únicos y no representan "
            "el riesgo recurrente que debe planificarse."
        ),
        limitation=(
            "Los valores provienen de un reanálisis en malla regular, no de estaciones "
            "meteorológicas puntuales: representan el promedio de una celda de ≈ 9 km y suavizan "
            "los extremos locales, en particular en fondos de valle y laderas."
        ),
    )


def interpret_time_series(frame: pd.DataFrame, *, frost_column: str = "helada") -> SectionInterpretation:
    """Redacta la interpretación del análisis de evolución temporal.

    Args:
        frame: Subconjunto filtrado.
        frost_column: Columna booleana de helada del umbral activo.

    Returns:
        Interpretación con los cinco puntos que exige la consigna.
    """
    annual = aggregate_annual(frame, frost_column=frost_column)
    if len(annual) < 3:
        return SectionInterpretation(
            represents="Evolución interanual de la incidencia de heladas.",
            pattern="El rango seleccionado abarca muy pocos años para evaluar una tendencia.",
            finding="Se requieren al menos tres años completos.",
            decision="Amplíe el rango de fechas.",
            limitation="Una serie corta no permite separar tendencia de variabilidad natural.",
        )

    test = mann_kendall_test(annual["heladas_por_provincia"])
    trend = fit_linear_trend(annual["anio"], annual["heladas_por_provincia"])

    peak_year = int(annual.loc[annual["heladas_por_provincia"].idxmax(), "anio"])
    peak_value = float(annual["heladas_por_provincia"].max())
    low_year = int(annual.loc[annual["heladas_por_provincia"].idxmin(), "anio"])
    low_value = float(annual["heladas_por_provincia"].min())
    variability = float(annual["heladas_por_provincia"].std())

    if test is not None and test.is_significant:
        pattern = (
            f"La prueba de Mann-Kendall detecta una tendencia {test.trend} significativa "
            f"({format_pvalue(test.p_value)}), con una pendiente de Sen de "
            f"{format_number(test.sen_slope, decimals=2, signed=True)} días por año."
        )
    else:
        pattern = (
            "No se detecta tendencia monótona significativa "
            f"({format_pvalue(test.p_value) if test else 'p = —'}); domina la variabilidad interanual."
        )

    if trend is not None:
        pattern += (
            f" El ajuste lineal explica el {format_percent(trend.r_squared * 100)} de la varianza "
            "interanual."
        )

    return SectionInterpretation(
        represents=(
            f"Número medio de días con helada por provincia en cada uno de los "
            f"{len(annual)} años del periodo seleccionado."
        ),
        pattern=pattern,
        finding=(
            f"El año más adverso fue **{peak_year}**, con {format_number(peak_value, decimals=1)} "
            f"días de helada por provincia, y el más benigno **{low_year}**, con "
            f"{format_number(low_value, decimals=1)}. La diferencia entre ambos "
            f"({format_number(peak_value - low_value, decimals=1)} días) supera con holgura la "
            f"desviación típica interanual ({format_number(variability, decimals=1)} días), lo que "
            "confirma que la variabilidad entre años es el rasgo dominante de la serie."
        ),
        decision=(
            "Diseñar los instrumentos de gestión del riesgo —seguro agrario, fondos de "
            "contingencia— para absorber la variabilidad interanual observada, que es el riesgo "
            "efectivo, en lugar de calibrarlos sobre el promedio del periodo."
        ),
        limitation=(
            "Diez años son suficientes para caracterizar la variabilidad interanual, pero "
            "insuficientes para atribuir cambios al calentamiento global: la Organización "
            "Meteorológica Mundial exige series de treinta años para establecer normales "
            "climáticas y detectar señales de cambio climático."
        ),
    )


def interpret_multivariate(
    frame: pd.DataFrame, pca_result: object | None, clustering_result: object | None
) -> SectionInterpretation:
    """Redacta la interpretación del análisis multidimensional.

    Args:
        frame: Subconjunto filtrado.
        pca_result: Resultado del PCA, o ``None``.
        clustering_result: Resultado del agrupamiento, o ``None``.

    Returns:
        Interpretación con los cinco puntos que exige la consigna.
    """
    if pca_result is None:
        return SectionInterpretation(
            represents="Estructura latente del sistema agroclimático regional.",
            pattern="La selección no contiene observaciones suficientes para el análisis.",
            finding="Se requieren al menos treinta observaciones completas.",
            decision="Amplíe el rango de filtros.",
            limitation="Un subconjunto reducido no permite estimar la matriz de covarianzas.",
        )

    variance_2d = float(pca_result.explained_variance[:2].sum() * 100)  # type: ignore[attr-defined]
    n_90 = pca_result.n_components_90  # type: ignore[attr-defined]
    n_features = len(pca_result.features)  # type: ignore[attr-defined]

    pattern = (
        f"Las dos primeras componentes principales concentran el "
        f"{format_percent(variance_2d)} de la varianza total de {n_features} variables "
        f"originales, y se necesitan {n_90} componentes para alcanzar el 90 %. "
        f"{pca_result.interpret_component(0)} {pca_result.interpret_component(1)}"  # type: ignore[attr-defined]
    )

    if clustering_result is not None:
        pattern += (
            f" El agrupamiento por K-Means identifica "
            f"{clustering_result.n_clusters} regímenes agroclimáticos diferenciados, con un "  # type: ignore[attr-defined]
            f"coeficiente de silueta de "
            f"{format_number(clustering_result.silhouette, decimals=3, thousands=False)} "  # type: ignore[attr-defined]
            f"({clustering_result.quality})."  # type: ignore[attr-defined]
        )

    return SectionInterpretation(
        represents=(
            f"Reducción de {n_features} variables agroclimáticas a un espacio de pocas "
            "dimensiones, donde la proximidad entre puntos significa semejanza climática."
        ),
        pattern=pattern,
        finding=(
            "La alta concentración de varianza en pocas componentes demuestra que el sistema "
            "agroclimático del altiplano está gobernado por un número reducido de factores "
            "latentes —esencialmente un eje térmico y un eje de humedad—, y no por la variación "
            "independiente de cada variable medida."
        ),
        decision=(
            "Basar el monitoreo operativo en las pocas variables que dominan las primeras "
            "componentes: se conserva la mayor parte de la información con una fracción del "
            "costo de instrumentación y mantenimiento."
        ),
        limitation=(
            "El análisis de componentes principales sólo capta relaciones lineales y es sensible "
            "a la escala, por lo que exige la estandarización previa que aquí se aplica. Además, "
            "los agrupamientos describen el periodo observado y no constituyen una zonificación "
            "normativa oficial."
        ),
    )


def interpret_geographic(province_summary: pd.DataFrame) -> SectionInterpretation:
    """Redacta la interpretación del análisis geográfico.

    Args:
        province_summary: Resumen por provincia.

    Returns:
        Interpretación con los cinco puntos que exige la consigna.
    """
    if province_summary.empty:
        return SectionInterpretation(
            represents="Distribución territorial del riesgo por heladas.",
            pattern="Sin provincias en la selección actual.",
            finding="No hay información territorial disponible.",
            decision="Amplíe el filtro de provincias.",
            limitation="La selección activa excluye todas las provincias.",
        )

    top = province_summary.iloc[0]
    n_high_risk = int((province_summary["tasa_helada"] >= 30).sum())
    basin_summary = (
        province_summary.groupby("cuenca", observed=True)["tasa_helada"].mean()
        if "cuenca" in province_summary.columns
        else pd.Series(dtype="float64")
    )

    pattern = (
        f"De las {len(province_summary)} provincias analizadas, {n_high_risk} superan el 30 % de "
        f"días con helada. **{top['provincia']}** lidera el ranking con "
        f"{format_percent(top['tasa_helada'])}, a "
        f"{format_number(top['altitud'], decimals=0, unit='msnm')}."
    )
    if len(basin_summary) > 1:
        highest = basin_summary.idxmax()
        pattern += (
            f" Por vertiente hidrográfica, la cuenca **{highest}** presenta la mayor incidencia "
            f"media ({format_percent(float(basin_summary.max()))}) frente a "
            f"{format_percent(float(basin_summary.min()))} en la otra."
        )

    return SectionInterpretation(
        represents=(
            "Localización geográfica de cada capital provincial, con el tamaño y el color de "
            "cada marca codificando la magnitud del riesgo agroclimático."
        ),
        pattern=pattern,
        finding=(
            "El riesgo no se distribuye al azar sobre el territorio: se organiza según la "
            "altitud y la proximidad al lago Titicaca, cuya inercia térmica atenúa el "
            "enfriamiento nocturno en las provincias circunlacustres."
        ),
        decision=(
            "Priorizar la inversión en infraestructura de protección y en estaciones "
            "meteorológicas en las provincias del extremo superior del ranking, donde el "
            "beneficio marginal por unidad invertida es mayor."
        ),
        limitation=(
            "Cada provincia está representada por un único punto —su capital—, de modo que el "
            "mapa no refleja la variabilidad interna del territorio provincial, que en zonas de "
            "topografía abrupta como Carabaya o Sandia puede ser considerable."
        ),
    )
