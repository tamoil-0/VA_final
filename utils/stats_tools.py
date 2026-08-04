"""Herramientas estadísticas: tendencia, correlación y contraste de hipótesis.

Todas las pruebas se implementan sobre NumPy y SciPy, sin dependencias
adicionales. Cada resultado se devuelve en una estructura de datos que incluye
el estadístico, su valor p y una **interpretación redactada**, de modo que la
capa de presentación no tenga que decidir cómo se lee un resultado: esa decisión
es estadística, no visual, y pertenece a este módulo.

Criterio metodológico sobre la tendencia
----------------------------------------
La tendencia se estima con **dos** métodos complementarios, no con uno:

* **Regresión lineal por mínimos cuadrados (OLS)**, que es óptima si los
  residuos son aproximadamente normales pero muy sensible a valores atípicos.
* **Prueba de Mann-Kendall con pendiente de Sen**, no paramétrica, que no supone
  distribución alguna y resiste los atípicos.

Reportar ambas es lo honesto: si coinciden, la conclusión es robusta; si
discrepan, la discrepancia es en sí misma un hallazgo que el informe debe
declarar en lugar de ocultar eligiendo el método más favorable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

#: Nivel de significancia adoptado en todo el proyecto.
ALPHA: float = 0.05


# ---------------------------------------------------------------------------
# Regresión lineal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearTrend:
    """Resultado de un ajuste lineal por mínimos cuadrados.

    Attributes:
        slope: Pendiente estimada, en unidades de ``y`` por unidad de ``x``.
        intercept: Ordenada al origen.
        r_squared: Proporción de varianza explicada.
        p_value: Valor p de la hipótesis nula de pendiente cero.
        standard_error: Error estándar de la pendiente.
        n: Número de observaciones empleadas.
        confidence_interval: Intervalo de confianza al 95 % de la pendiente.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    standard_error: float
    n: int
    confidence_interval: tuple[float, float]

    @property
    def is_significant(self) -> bool:
        """Indica si la pendiente difiere de cero al nivel :data:`ALPHA`."""
        return bool(self.p_value < ALPHA)

    @property
    def direction(self) -> Literal["creciente", "decreciente", "estable"]:
        """Sentido de la tendencia, ``estable`` si no es significativa."""
        if not self.is_significant:
            return "estable"
        return "creciente" if self.slope > 0 else "decreciente"

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evalúa la recta ajustada.

        Args:
            x: Valores de la variable independiente.

        Returns:
            Valores ajustados de la variable dependiente.
        """
        return self.intercept + self.slope * np.asarray(x, dtype="float64")


def fit_linear_trend(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> LinearTrend | None:
    """Ajusta una recta por mínimos cuadrados y evalúa su significancia.

    Args:
        x: Variable independiente.
        y: Variable dependiente.

    Returns:
        El resultado del ajuste, o ``None`` si hay menos de tres pares completos
        (con menos observaciones la pendiente no es estimable con error).
    """
    frame = pd.DataFrame({"x": np.asarray(x, dtype="float64"), "y": np.asarray(y, dtype="float64")}).dropna()
    if len(frame) < 3:
        return None

    result = stats.linregress(frame["x"], frame["y"])
    # Intervalo de confianza de la pendiente basado en la t de Student, con
    # n - 2 grados de libertad (se estiman dos parámetros).
    critical = stats.t.ppf(1 - ALPHA / 2, df=len(frame) - 2)
    margin = critical * result.stderr

    return LinearTrend(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue**2),
        p_value=float(result.pvalue),
        standard_error=float(result.stderr),
        n=int(len(frame)),
        confidence_interval=(float(result.slope - margin), float(result.slope + margin)),
    )


# ---------------------------------------------------------------------------
# Mann-Kendall y pendiente de Sen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MannKendallResult:
    """Resultado de la prueba de tendencia monótona de Mann-Kendall.

    Attributes:
        statistic_s: Estadístico S de Mann-Kendall.
        z_score: Estadístico normalizado.
        p_value: Valor p bilateral.
        tau: Coeficiente tau de Kendall.
        sen_slope: Pendiente de Sen (mediana de las pendientes por pares).
        trend: Sentido de la tendencia detectada.
        n: Número de observaciones.
    """

    statistic_s: float
    z_score: float
    p_value: float
    tau: float
    sen_slope: float
    trend: Literal["creciente", "decreciente", "sin tendencia"]
    n: int

    @property
    def is_significant(self) -> bool:
        """Indica si se rechaza la hipótesis nula de ausencia de tendencia."""
        return bool(self.p_value < ALPHA)


def mann_kendall_test(values: pd.Series | np.ndarray) -> MannKendallResult | None:
    """Aplica la prueba de Mann-Kendall con estimación de la pendiente de Sen.

    La implementación incluye la **corrección por empates** en la varianza del
    estadístico, indispensable con series de conteos como «días de helada por
    año», donde los valores repetidos son frecuentes. Omitirla subestimaría la
    varianza y produciría valores p artificialmente pequeños.

    Args:
        values: Serie ordenada cronológicamente.

    Returns:
        El resultado de la prueba, o ``None`` si hay menos de cuatro
        observaciones (por debajo de ese tamaño la aproximación normal no aplica).
    """
    series = pd.Series(values, dtype="float64").dropna().reset_index(drop=True)
    n = len(series)
    if n < 4:
        return None

    data = series.to_numpy()

    # Estadístico S: suma de los signos de todas las diferencias por pares.
    statistic_s = float(
        sum(np.sign(data[j] - data[i]).sum() for i, j in [(i, slice(i + 1, n)) for i in range(n - 1)])
    )

    # Varianza con corrección por empates.
    _, tie_counts = np.unique(data, return_counts=True)
    tie_correction = float(sum(t * (t - 1) * (2 * t + 5) for t in tie_counts if t > 1))
    variance = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18.0

    if variance <= 0:
        z_score = 0.0
    elif statistic_s > 0:
        z_score = (statistic_s - 1) / np.sqrt(variance)
    elif statistic_s < 0:
        z_score = (statistic_s + 1) / np.sqrt(variance)
    else:
        z_score = 0.0

    p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))
    tau = statistic_s / (0.5 * n * (n - 1)) if n > 1 else 0.0

    # Pendiente de Sen: mediana de las pendientes de todos los pares posibles.
    pairwise_slopes = [
        (data[j] - data[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)
    ]
    sen_slope = float(np.median(pairwise_slopes)) if pairwise_slopes else 0.0

    if p_value < ALPHA:
        trend: Literal["creciente", "decreciente", "sin tendencia"] = (
            "creciente" if statistic_s > 0 else "decreciente"
        )
    else:
        trend = "sin tendencia"

    return MannKendallResult(
        statistic_s=statistic_s,
        z_score=float(z_score),
        p_value=p_value,
        tau=float(tau),
        sen_slope=sen_slope,
        trend=trend,
        n=n,
    )


# ---------------------------------------------------------------------------
# Correlación
# ---------------------------------------------------------------------------


def correlation_matrix(
    frame: pd.DataFrame, variables: list[str], *, method: Literal["pearson", "spearman"] = "pearson"
) -> pd.DataFrame:
    """Calcula la matriz de correlación entre las variables indicadas.

    Args:
        frame: Dataframe de origen.
        variables: Columnas numéricas a correlacionar.
        method: ``"pearson"`` para relaciones lineales, ``"spearman"`` para
            relaciones monótonas no necesariamente lineales.

    Returns:
        Matriz cuadrada de coeficientes de correlación.
    """
    available = [variable for variable in variables if variable in frame.columns]
    return frame.loc[:, available].corr(method=method)


def correlation_pvalues(frame: pd.DataFrame, variables: list[str], *, method: str = "pearson") -> pd.DataFrame:
    """Calcula la matriz de valores p asociada a la matriz de correlación.

    Un coeficiente sin su valor p es incompleto: con 47 000 observaciones,
    correlaciones de magnitud trivial resultan «significativas», y sin la matriz
    de valores p el lector no puede distinguir una relación fuerte de una
    meramente detectable.

    Args:
        frame: Dataframe de origen.
        variables: Columnas numéricas a correlacionar.
        method: ``"pearson"`` o ``"spearman"``.

    Returns:
        Matriz cuadrada de valores p, con ceros en la diagonal.
    """
    available = [variable for variable in variables if variable in frame.columns]
    data = frame.loc[:, available].dropna()
    size = len(available)
    matrix = pd.DataFrame(np.zeros((size, size)), index=available, columns=available)

    test = stats.pearsonr if method == "pearson" else stats.spearmanr
    for i, first in enumerate(available):
        for j, second in enumerate(available):
            if i >= j:
                continue
            if len(data) < 3:
                p_value = float("nan")
            else:
                p_value = float(test(data[first], data[second])[1])
            matrix.iloc[i, j] = p_value
            matrix.iloc[j, i] = p_value
    return matrix


@dataclass(frozen=True)
class CorrelationFinding:
    """Par de variables con correlación relevante.

    Attributes:
        first: Nombre de la primera variable.
        second: Nombre de la segunda variable.
        coefficient: Coeficiente de correlación.
        p_value: Valor p de la prueba de correlación nula.
        strength: Calificación verbal de la magnitud.
    """

    first: str
    second: str
    coefficient: float
    p_value: float
    strength: str

    @property
    def direction(self) -> Literal["positiva", "negativa"]:
        """Sentido de la asociación."""
        return "positiva" if self.coefficient > 0 else "negativa"


def classify_correlation_strength(coefficient: float) -> str:
    """Clasifica verbalmente la magnitud de una correlación.

    Se emplean los cortes convencionales en ciencias ambientales (Cohen, 1988).

    Args:
        coefficient: Coeficiente de correlación.

    Returns:
        Calificación verbal de la magnitud.
    """
    magnitude = abs(coefficient)
    if magnitude >= 0.9:
        return "casi perfecta"
    if magnitude >= 0.7:
        return "fuerte"
    if magnitude >= 0.5:
        return "moderada"
    if magnitude >= 0.3:
        return "débil"
    return "muy débil"


def top_correlations(
    frame: pd.DataFrame,
    variables: list[str],
    *,
    target: str | None = None,
    limit: int = 8,
    method: Literal["pearson", "spearman"] = "pearson",
) -> list[CorrelationFinding]:
    """Extrae los pares de variables con mayor correlación absoluta.

    Args:
        frame: Dataframe de origen.
        variables: Columnas numéricas a considerar.
        target: Si se indica, sólo se devuelven pares que involucren a esta
            variable; es el caso de uso habitual al explicar un fenómeno concreto.
        limit: Número máximo de pares a devolver.
        method: Método de correlación.

    Returns:
        Lista de hallazgos ordenada por magnitud descendente.
    """
    coefficients = correlation_matrix(frame, variables, method=method)
    p_values = correlation_pvalues(frame, list(coefficients.columns), method=method)

    findings: list[CorrelationFinding] = []
    columns = list(coefficients.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            if target is not None and target not in (first, second):
                continue
            coefficient = float(coefficients.loc[first, second])
            if pd.isna(coefficient):
                continue
            findings.append(
                CorrelationFinding(
                    first=first,
                    second=second,
                    coefficient=coefficient,
                    p_value=float(p_values.loc[first, second]),
                    strength=classify_correlation_strength(coefficient),
                )
            )

    findings.sort(key=lambda finding: abs(finding.coefficient), reverse=True)
    return findings[:limit]


def variance_inflation_factors(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Calcula el factor de inflación de la varianza de cada variable.

    El VIF cuantifica la multicolinealidad: mide cuánto se infla la varianza del
    coeficiente de una variable por su dependencia lineal con las demás. Es la
    comprobación que justifica —o desaconseja— interpretar los pesos de las
    componentes principales variable por variable.

    Args:
        frame: Dataframe de origen.
        variables: Columnas numéricas a evaluar.

    Returns:
        Dataframe con el VIF y su lectura, ordenado de mayor a menor.
    """
    available = [variable for variable in variables if variable in frame.columns]
    data = frame.loc[:, available].dropna()

    rows: list[dict[str, object]] = []
    for variable in available:
        others = [column for column in available if column != variable]
        if not others or len(data) < len(available) + 2:
            continue

        design = np.column_stack([np.ones(len(data)), data[others].to_numpy()])
        target = data[variable].to_numpy()
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals = target - design @ coefficients

        total_variance = float(((target - target.mean()) ** 2).sum())
        r_squared = 1.0 - float((residuals**2).sum()) / total_variance if total_variance > 0 else 0.0
        vif = float("inf") if r_squared >= 0.9999 else 1.0 / (1.0 - r_squared)

        if vif >= 10:
            reading = "multicolinealidad severa"
        elif vif >= 5:
            reading = "multicolinealidad moderada"
        else:
            reading = "aceptable"

        rows.append({"variable": variable, "VIF": round(vif, 2), "lectura": reading})

    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Contrastes de hipótesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupComparison:
    """Resultado de comparar una variable numérica entre grupos.

    Attributes:
        test_name: Nombre de la prueba aplicada.
        statistic: Valor del estadístico de contraste.
        p_value: Valor p de la prueba.
        n_groups: Número de grupos comparados.
        effect_size: Tamaño del efecto (eta cuadrado épsilon).
        interpretation: Lectura redactada del resultado.
    """

    test_name: str
    statistic: float
    p_value: float
    n_groups: int
    effect_size: float
    interpretation: str

    @property
    def is_significant(self) -> bool:
        """Indica si las diferencias entre grupos son significativas."""
        return bool(self.p_value < ALPHA)


def compare_groups(frame: pd.DataFrame, value_column: str, group_column: str) -> GroupComparison | None:
    """Contrasta si una variable numérica difiere entre grupos.

    Se emplea la prueba de **Kruskal-Wallis**, no un ANOVA, por una razón
    sustantiva: las series climáticas diarias no son normales —la precipitación
    tiene una masa de probabilidad concentrada en cero y las temperaturas mínimas
    presentan asimetría marcada—, y el ANOVA exige normalidad y homocedasticidad.
    Kruskal-Wallis sólo requiere independencia y ordenación.

    Args:
        frame: Dataframe de origen.
        value_column: Variable numérica a comparar.
        group_column: Variable categórica que define los grupos.

    Returns:
        El resultado del contraste, o ``None`` si no hay al menos dos grupos con
        datos suficientes.
    """
    if value_column not in frame.columns or group_column not in frame.columns:
        return None

    groups = [
        group[value_column].dropna().to_numpy()
        for _, group in frame.groupby(group_column, observed=True)
        if group[value_column].notna().sum() >= 3
    ]
    if len(groups) < 2:
        return None

    statistic, p_value = stats.kruskal(*groups)
    total = sum(len(group) for group in groups)

    # Eta cuadrado épsilon: tamaño del efecto asociado a Kruskal-Wallis.
    effect_size = (
        max(0.0, (float(statistic) - len(groups) + 1) / (total - len(groups)))
        if total > len(groups)
        else 0.0
    )

    if p_value < ALPHA:
        magnitude = "grande" if effect_size >= 0.14 else "medio" if effect_size >= 0.06 else "pequeño"
        interpretation = (
            f"Las diferencias entre los {len(groups)} grupos son estadísticamente "
            f"significativas, con un tamaño de efecto {magnitude} (ε² = {effect_size:.3f})."
        )
    else:
        interpretation = (
            f"No se detectan diferencias significativas entre los {len(groups)} grupos "
            f"al nivel de significancia del {ALPHA:.0%}."
        )

    return GroupComparison(
        test_name="Kruskal-Wallis",
        statistic=float(statistic),
        p_value=float(p_value),
        n_groups=len(groups),
        effect_size=float(effect_size),
        interpretation=interpretation,
    )


def describe_distribution(series: pd.Series) -> dict[str, float]:
    """Calcula un resumen descriptivo completo de una variable numérica.

    Args:
        series: Serie numérica a describir.

    Returns:
        Diccionario con medidas de posición, dispersión y forma. Se incluyen la
        asimetría y la curtosis porque determinan si las medidas centradas en la
        media son representativas o si debe preferirse la mediana.
    """
    clean = pd.Series(series).dropna().astype("float64")
    if clean.empty:
        return {}

    quartile_1, quartile_3 = clean.quantile([0.25, 0.75])
    return {
        "n": float(len(clean)),
        "media": float(clean.mean()),
        "mediana": float(clean.median()),
        "desv_estandar": float(clean.std()),
        "minimo": float(clean.min()),
        "maximo": float(clean.max()),
        "rango": float(clean.max() - clean.min()),
        "q1": float(quartile_1),
        "q3": float(quartile_3),
        "rango_interquartil": float(quartile_3 - quartile_1),
        "asimetria": float(clean.skew()),
        "curtosis": float(clean.kurtosis()),
        "coef_variacion": float(clean.std() / clean.mean()) if clean.mean() != 0 else float("nan"),
        "p05": float(clean.quantile(0.05)),
        "p95": float(clean.quantile(0.95)),
    }


def modified_z_scores(series: pd.Series) -> pd.Series:
    """Calcula la puntuación Z modificada, basada en la mediana y la MAD.

    Se prefiere a la puntuación Z clásica porque la media y la desviación típica
    están contaminadas por los propios valores atípicos que se busca detectar,
    mientras que la mediana y la desviación absoluta mediana no lo están.

    Args:
        series: Serie numérica.

    Returns:
        Serie con las puntuaciones Z modificadas. El criterio convencional
        (Iglewicz y Hoaglin, 1993) considera atípico un valor absoluto > 3,5.
    """
    values = pd.Series(series).astype("float64")
    median = values.median()
    absolute_deviation = (values - median).abs().median()

    if absolute_deviation == 0 or pd.isna(absolute_deviation):
        return pd.Series(np.zeros(len(values)), index=values.index)

    # La constante 0,6745 es el cuantil 0,75 de la normal estándar; escala la MAD
    # para que sea comparable con una desviación típica.
    return 0.6745 * (values - median) / absolute_deviation
