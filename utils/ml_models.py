"""Técnicas avanzadas: reducción de dimensionalidad, agrupamiento y anomalías.

Este módulo implementa las técnicas que la consigna clasifica como «avanzadas».
Se incluyen todas ellas, no una sola, porque cada una responde a una faceta
distinta de la pregunta P6:

============================  ====================================================
Técnica                       Qué pregunta responde
============================  ====================================================
PCA                           ¿Cuántos factores latentes gobiernan el clima y
                              cuáles son?
K-Means                       ¿Existen zonas agroclimáticas homogéneas?
Clustering jerárquico         ¿Cómo se anidan esas zonas entre sí?
Isolation Forest              ¿Qué días son atípicos considerando todas las
                              variables a la vez?
Z modificada (MAD)            ¿Qué días son extremos en la variable central?
Random Forest                 ¿Qué variables predicen mejor una helada y con
                              qué capacidad discriminante?
============================  ====================================================

Reproducibilidad
----------------
Todo procedimiento estocástico recibe :data:`config.settings.RANDOM_STATE`. Dos
ejecuciones sobre los mismos datos producen resultados idénticos, condición
indispensable para que las cifras del informe coincidan con las del dashboard.

Estandarización
---------------
Las variables se estandarizan **siempre** antes de PCA, K-Means y clustering
jerárquico. Sin ello, la altitud (miles de msnm) dominaría por completo la
varianza total frente a la temperatura (decenas de °C), y las componentes
resultantes describirían la escala de medida en lugar del fenómeno.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import RANDOM_STATE
from utils.formatting import variable_label


def prepare_matrix(
    frame: pd.DataFrame, features: list[str], *, standardize: bool = True
) -> tuple[np.ndarray, list[str], pd.Index]:
    """Extrae y estandariza la matriz de características.

    Args:
        frame: Dataframe de origen.
        features: Variables a incluir. Se descartan silenciosamente las que no
            existan o sean constantes en el subconjunto: una variable constante
            tiene varianza cero y haría fallar la estandarización.
        standardize: Si se centra y escala cada variable.

    Returns:
        Tupla ``(matriz, variables_efectivas, índice_de_filas)``.
    """
    available = [
        feature
        for feature in features
        if feature in frame.columns and frame[feature].notna().any() and frame[feature].nunique() > 1
    ]
    if not available:
        return np.empty((0, 0)), [], pd.Index([])

    data = frame.loc[:, available].dropna()
    if data.empty:
        return np.empty((0, 0)), available, pd.Index([])

    matrix = data.to_numpy(dtype="float64")
    if standardize:
        matrix = StandardScaler().fit_transform(matrix)
    return matrix, available, data.index


# ---------------------------------------------------------------------------
# Análisis de componentes principales
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PcaResult:
    """Resultado de un análisis de componentes principales.

    Attributes:
        scores: Coordenadas de las observaciones en el espacio de componentes.
        loadings: Pesos de cada variable en cada componente.
        explained_variance: Proporción de varianza explicada por componente.
        cumulative_variance: Varianza explicada acumulada.
        features: Variables efectivamente empleadas.
        row_index: Índice de las filas que entraron en el análisis.
        n_components_90: Componentes necesarias para explicar el 90 % de la varianza.
    """

    scores: np.ndarray
    loadings: pd.DataFrame
    explained_variance: np.ndarray
    cumulative_variance: np.ndarray
    features: list[str]
    row_index: pd.Index
    n_components_90: int

    def component_label(self, index: int) -> str:
        """Etiqueta de una componente con su varianza explicada.

        Args:
            index: Índice de la componente, empezando en cero.

        Returns:
            Etiqueta del tipo ``«CP1 (42,3 % de la varianza)»``.
        """
        share = self.explained_variance[index] * 100.0
        return f"CP{index + 1} ({share:.1f} % de la varianza)".replace(".", ",")

    def dominant_variables(self, component: int, *, top: int = 3) -> list[tuple[str, float]]:
        """Variables con mayor peso absoluto en una componente.

        Args:
            component: Índice de la componente, empezando en cero.
            top: Número de variables a devolver.

        Returns:
            Lista de pares ``(variable, peso)`` ordenada por peso absoluto.
        """
        column = self.loadings.iloc[:, component]
        ordered = column.reindex(column.abs().sort_values(ascending=False).index)
        return [(str(name), float(value)) for name, value in ordered.head(top).items()]

    def interpret_component(self, component: int) -> str:
        """Redacta la interpretación sustantiva de una componente.

        Args:
            component: Índice de la componente, empezando en cero.

        Returns:
            Frase que nombra el eje latente en términos del dominio.
        """
        dominant = self.dominant_variables(component, top=3)
        positive = [variable_label(name, with_unit=False) for name, weight in dominant if weight > 0]
        negative = [variable_label(name, with_unit=False) for name, weight in dominant if weight < 0]

        parts: list[str] = []
        if positive:
            parts.append("aumenta con " + ", ".join(positive))
        if negative:
            parts.append("disminuye con " + ", ".join(negative))
        return f"{self.component_label(component)}: " + " y ".join(parts) + "."


def run_pca(frame: pd.DataFrame, features: list[str], *, n_components: int = 5) -> PcaResult | None:
    """Ejecuta un análisis de componentes principales sobre datos estandarizados.

    Args:
        frame: Dataframe de origen.
        features: Variables a incluir en el análisis.
        n_components: Número máximo de componentes a retener.

    Returns:
        El resultado del análisis, o ``None`` si no hay datos suficientes para
        estimar la matriz de covarianzas con fiabilidad.
    """
    matrix, effective, row_index = prepare_matrix(frame, features)
    if matrix.size == 0 or matrix.shape[0] < max(30, matrix.shape[1] + 2):
        return None

    components = min(n_components, matrix.shape[1], matrix.shape[0])
    model = PCA(n_components=components, random_state=RANDOM_STATE)
    scores = model.fit_transform(matrix)

    loadings = pd.DataFrame(
        model.components_.T,
        index=effective,
        columns=[f"CP{index + 1}" for index in range(components)],
    )
    cumulative = np.cumsum(model.explained_variance_ratio_)

    return PcaResult(
        scores=scores,
        loadings=loadings,
        explained_variance=model.explained_variance_ratio_,
        cumulative_variance=cumulative,
        features=effective,
        row_index=row_index,
        n_components_90=int(np.searchsorted(cumulative, 0.90) + 1),
    )


# ---------------------------------------------------------------------------
# Agrupamiento
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusteringResult:
    """Resultado de un agrupamiento por K-Means.

    Attributes:
        labels: Etiqueta de grupo asignada a cada observación.
        n_clusters: Número de grupos formados.
        silhouette: Coeficiente de silueta del agrupamiento elegido.
        inertia: Suma de distancias cuadradas intra-grupo.
        centroids: Centroides expresados en las unidades originales.
        profile: Perfil descriptivo de cada grupo.
        row_index: Índice de las filas agrupadas.
        diagnostics: Silueta e inercia para cada ``k`` evaluado.
        features: Variables empleadas.
    """

    labels: np.ndarray
    n_clusters: int
    silhouette: float
    inertia: float
    centroids: pd.DataFrame
    profile: pd.DataFrame
    row_index: pd.Index
    diagnostics: pd.DataFrame
    features: list[str]

    @property
    def quality(self) -> str:
        """Lectura cualitativa del coeficiente de silueta."""
        if self.silhouette >= 0.7:
            return "estructura fuerte"
        if self.silhouette >= 0.5:
            return "estructura razonable"
        if self.silhouette >= 0.25:
            return "estructura débil"
        return "sin estructura clara"


def select_optimal_k(
    matrix: np.ndarray, *, k_range: range = range(2, 8), sample_size: int = 5_000
) -> pd.DataFrame:
    """Evalúa la calidad del agrupamiento para varios valores de ``k``.

    El coeficiente de silueta se calcula sobre una submuestra aleatoria porque su
    coste es cuadrático en el número de observaciones: con 47 000 filas exigiría
    más de mil millones de distancias por cada ``k``. La submuestra es fija por la
    semilla, de modo que la selección sigue siendo reproducible.

    Args:
        matrix: Matriz de características ya estandarizada.
        k_range: Valores de ``k`` a evaluar.
        sample_size: Tamaño de la submuestra usada en la silueta.

    Returns:
        Dataframe con la inercia y la silueta de cada ``k``.
    """
    generator = np.random.default_rng(RANDOM_STATE)
    if len(matrix) > sample_size:
        indices = generator.choice(len(matrix), size=sample_size, replace=False)
        sample = matrix[indices]
    else:
        sample = matrix

    rows: list[dict[str, float]] = []
    for k in k_range:
        if k >= len(sample):
            continue
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(sample)
        rows.append(
            {
                "k": int(k),
                "inercia": float(model.inertia_),
                "silueta": float(silhouette_score(sample, labels)),
            }
        )
    return pd.DataFrame(rows)


def run_kmeans(
    frame: pd.DataFrame,
    features: list[str],
    *,
    n_clusters: int | None = None,
    max_clusters: int = 4,
) -> ClusteringResult | None:
    """Agrupa observaciones por K-Means y describe cada grupo formado.

    Args:
        frame: Dataframe de origen.
        features: Variables a emplear en el agrupamiento.
        n_clusters: Número de grupos. Si es ``None`` se elige el ``k`` que
            maximiza el coeficiente de silueta.
        max_clusters: Tope de grupos. Está limitado a cuatro de forma deliberada:
            los grupos se representan en diagramas de dispersión, donde cualquier
            par de marcas puede quedar adyacente, y la paleta verificada del
            proyecto garantiza la distinción de hasta cuatro series en esa
            condición.

    Returns:
        El resultado del agrupamiento, o ``None`` si no hay datos suficientes.
    """
    matrix, effective, row_index = prepare_matrix(frame, features)
    if matrix.size == 0 or len(matrix) < 50:
        return None

    diagnostics = select_optimal_k(matrix, k_range=range(2, max_clusters + 1))
    if diagnostics.empty:
        return None

    if n_clusters is None:
        n_clusters = int(diagnostics.loc[diagnostics["silueta"].idxmax(), "k"])
    n_clusters = max(2, min(n_clusters, max_clusters))

    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(matrix)

    generator = np.random.default_rng(RANDOM_STATE)
    if len(matrix) > 5_000:
        indices = generator.choice(len(matrix), size=5_000, replace=False)
        silhouette = float(silhouette_score(matrix[indices], labels[indices]))
    else:
        silhouette = float(silhouette_score(matrix, labels))

    # Los centroides se devuelven en unidades originales: un centroide en
    # unidades estandarizadas es ilegible para quien debe tomar decisiones.
    original = frame.loc[row_index, effective].copy()
    original["grupo"] = labels
    centroids = original.groupby("grupo")[effective].mean().round(2)

    profile = _build_cluster_profile(frame, row_index, labels)

    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        silhouette=silhouette,
        inertia=float(model.inertia_),
        centroids=centroids,
        profile=profile,
        row_index=row_index,
        diagnostics=diagnostics,
        features=effective,
    )


def _build_cluster_profile(frame: pd.DataFrame, row_index: pd.Index, labels: np.ndarray) -> pd.DataFrame:
    """Describe cada grupo en términos interpretables del dominio.

    Args:
        frame: Dataframe original completo.
        row_index: Índice de las filas agrupadas.
        labels: Etiquetas asignadas.

    Returns:
        Dataframe con una fila por grupo y su caracterización agroclimática.
    """
    context = frame.loc[row_index].copy()
    context["grupo"] = labels

    aggregations: dict[str, tuple[str, object]] = {
        "n_dias": ("fecha", "count"),
        "tmin_media": ("temperatura_minima", "mean"),
        "tmax_media": ("temperatura_maxima", "mean"),
        "oscilacion_media": ("oscilacion_termica", "mean"),
        "precipitacion_media": ("precipitacion", "mean"),
        "humedad_media": ("humedad_relativa", "mean"),
        "nubosidad_media": ("nubosidad", "mean"),
        "radiacion_media": ("radiacion_solar", "mean"),
        "viento_medio": ("velocidad_viento", "mean"),
        "altitud_media": ("altitud", "mean"),
        "tasa_helada": ("helada", "mean"),
        "riesgo_medio": ("indice_riesgo_helada", "mean"),
    }
    available = {
        name: specification
        for name, specification in aggregations.items()
        if specification[0] in context.columns
    }
    profile = context.groupby("grupo").agg(**available).reset_index()

    if "tasa_helada" in profile.columns:
        profile["tasa_helada"] = (profile["tasa_helada"] * 100.0).round(1)

    # Temporada y provincia predominantes: dan nombre reconocible al grupo.
    if "temporada" in context.columns:
        profile["temporada_dominante"] = (
            context.groupby("grupo")["temporada"]
            .agg(lambda series: series.mode().iat[0] if not series.mode().empty else "—")
            .to_numpy()
        )
    if "provincia" in context.columns:
        profile["provincias_frecuentes"] = (
            context.groupby("grupo")["provincia"]
            .agg(lambda series: ", ".join(series.value_counts().head(3).index.astype(str)))
            .to_numpy()
        )

    profile["etiqueta"] = [_name_cluster(row) for _, row in profile.iterrows()]
    return profile.round(2)


def _name_cluster(row: pd.Series) -> str:
    """Asigna un nombre descriptivo a un grupo según su perfil.

    Args:
        row: Fila del perfil de grupos.

    Returns:
        Nombre corto que resume el régimen agroclimático del grupo.
    """
    frost_rate = float(row.get("tasa_helada", 0.0))
    precipitation = float(row.get("precipitacion_media", 0.0))
    amplitude = float(row.get("oscilacion_media", 0.0))

    if frost_rate >= 60:
        severity = "Helada dominante"
    elif frost_rate >= 25:
        severity = "Helada frecuente"
    elif frost_rate >= 5:
        severity = "Helada ocasional"
    else:
        severity = "Libre de helada"

    humidity = "húmedo" if precipitation >= 3.0 else "seco"
    amplitude_label = "amplitud alta" if amplitude >= 14 else "amplitud moderada"
    return f"{severity} · {humidity} · {amplitude_label}"


@dataclass(frozen=True)
class HierarchicalResult:
    """Resultado de un agrupamiento jerárquico sobre perfiles provinciales.

    Attributes:
        linkage_matrix: Matriz de enlace en el formato de SciPy.
        labels: Etiqueta de grupo por provincia.
        province_names: Nombres de las provincias, en el orden de las filas.
        n_clusters: Número de grupos obtenidos al cortar el dendrograma.
    """

    linkage_matrix: np.ndarray
    labels: np.ndarray
    province_names: list[str]
    n_clusters: int


def run_hierarchical_clustering(
    province_summary: pd.DataFrame, features: list[str], *, n_clusters: int = 4
) -> HierarchicalResult | None:
    """Agrupa provincias jerárquicamente según su perfil agroclimático.

    Se aplica sobre los **perfiles provinciales agregados**, no sobre los días
    individuales: la pregunta es cómo se parecen los territorios entre sí, y el
    dendrograma resultante tiene trece hojas legibles en lugar de decenas de
    miles.

    Args:
        province_summary: Resumen por provincia.
        features: Variables del perfil a emplear.
        n_clusters: Número de grupos al cortar el dendrograma.

    Returns:
        El resultado del agrupamiento, o ``None`` si hay menos de tres provincias.
    """
    from scipy.cluster.hierarchy import linkage

    available = [feature for feature in features if feature in province_summary.columns]
    if len(province_summary) < 3 or not available:
        return None

    data = province_summary.loc[:, available].fillna(province_summary[available].mean())
    matrix = StandardScaler().fit_transform(data.to_numpy(dtype="float64"))

    # Enlace de Ward: minimiza el incremento de varianza intra-grupo en cada
    # fusión, criterio coherente con el uso de distancia euclídea.
    linkage_matrix = linkage(matrix, method="ward")
    effective_clusters = min(n_clusters, len(province_summary) - 1)
    labels = AgglomerativeClustering(n_clusters=effective_clusters, linkage="ward").fit_predict(matrix)

    return HierarchicalResult(
        linkage_matrix=linkage_matrix,
        labels=labels,
        province_names=province_summary["provincia"].astype(str).tolist(),
        n_clusters=effective_clusters,
    )


# ---------------------------------------------------------------------------
# Detección de anomalías
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyResult:
    """Resultado de la detección multivariante de anomalías.

    Attributes:
        flags: Serie booleana que marca las observaciones anómalas.
        scores: Puntuación de anomalía; los valores más bajos son más anómalos.
        n_anomalies: Número de observaciones marcadas.
        contamination: Proporción esperada de anomalías configurada.
        features: Variables empleadas.
        extremes: Las observaciones más anómalas, con su contexto.
    """

    flags: pd.Series
    scores: pd.Series
    n_anomalies: int
    contamination: float
    features: list[str]
    extremes: pd.DataFrame


def detect_anomalies(
    frame: pd.DataFrame, features: list[str], *, contamination: float = 0.01, top: int = 15
) -> AnomalyResult | None:
    """Detecta días atípicos con Isolation Forest.

    Se elige Isolation Forest y no un criterio por variable porque la pregunta es
    **multivariante**: un día puede no ser extremo en ninguna variable por
    separado y sí ser una combinación insólita —por ejemplo, temperatura mínima
    muy baja junto con humedad alta y cielo cubierto, combinación que contradice
    el mecanismo habitual de la helada de radiación—.

    Args:
        frame: Dataframe de origen.
        features: Variables a considerar.
        contamination: Proporción esperada de anomalías.
        top: Número de anomalías extremas a devolver con su contexto.

    Returns:
        El resultado de la detección, o ``None`` si no hay datos suficientes.
    """
    matrix, effective, row_index = prepare_matrix(frame, features)
    if matrix.size == 0 or len(matrix) < 100:
        return None

    model = IsolationForest(
        contamination=contamination, random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1
    )
    predictions = model.fit_predict(matrix)
    scores = model.score_samples(matrix)

    flags = pd.Series(False, index=frame.index)
    flags.loc[row_index] = predictions == -1

    score_series = pd.Series(np.nan, index=frame.index)
    score_series.loc[row_index] = scores

    context_columns = [
        column
        for column in (
            "fecha", "provincia", "temperatura_minima", "temperatura_maxima",
            "oscilacion_termica", "precipitacion", "humedad_relativa", "nubosidad",
            "radiacion_solar", "velocidad_viento", "indice_riesgo_helada",
            "intensidad_helada", "precipitacion_sospechosa",
        )
        if column in frame.columns
    ]
    extremes = (
        frame.loc[flags, context_columns]
        .assign(puntuacion_anomalia=score_series.loc[flags].round(4))
        .sort_values("puntuacion_anomalia")
        .head(top)
        .reset_index(drop=True)
    )

    return AnomalyResult(
        flags=flags,
        scores=score_series,
        n_anomalies=int(flags.sum()),
        contamination=contamination,
        features=effective,
        extremes=extremes,
    )


# ---------------------------------------------------------------------------
# Modelo predictivo e importancia de variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifierResult:
    """Resultado del modelo de clasificación de heladas.

    Attributes:
        importances: Importancia por permutación de cada variable.
        roc_auc: Área bajo la curva ROC en el conjunto de prueba.
        average_precision: Precisión media (área bajo la curva PR).
        confusion: Matriz de confusión en el conjunto de prueba.
        roc_curve_points: Puntos ``(fpr, tpr)`` de la curva ROC.
        pr_curve_points: Puntos ``(recall, precision)`` de la curva PR.
        accuracy: Exactitud global.
        recall: Sensibilidad sobre la clase positiva (helada).
        precision: Precisión sobre la clase positiva.
        n_train: Tamaño del conjunto de entrenamiento.
        n_test: Tamaño del conjunto de prueba.
        positive_rate: Prevalencia de la clase positiva.
        features: Variables empleadas.
    """

    importances: pd.DataFrame
    roc_auc: float
    average_precision: float
    confusion: np.ndarray
    roc_curve_points: tuple[np.ndarray, np.ndarray]
    pr_curve_points: tuple[np.ndarray, np.ndarray]
    accuracy: float
    recall: float
    precision: float
    n_train: int
    n_test: int
    positive_rate: float
    features: list[str] = field(default_factory=list)


def train_frost_classifier(
    frame: pd.DataFrame,
    features: list[str],
    *,
    target: str = "helada",
    test_size: float = 0.25,
) -> ClassifierResult | None:
    """Entrena un Random Forest para predecir la ocurrencia de helada.

    El propósito es **explicativo antes que predictivo**: se busca ordenar las
    variables por su contribución a la discriminación, no desplegar un sistema de
    pronóstico. Por eso se reporta la importancia por **permutación** y no la
    importancia por impureza de Gini: esta última está sesgada a favor de las
    variables con muchos valores distintos, mientras que la primera mide la
    pérdida real de capacidad predictiva al destruir la información de cada
    variable.

    La partición es **cronológica**, no aleatoria: las series climáticas están
    autocorrelacionadas, y una partición aleatoria dejaría días contiguos a ambos
    lados de la frontera, filtrando información del conjunto de prueba al de
    entrenamiento e inflando artificialmente las métricas.

    Args:
        frame: Dataframe de origen, ordenado o no.
        features: Variables predictoras.
        target: Variable objetivo booleana.
        test_size: Proporción final de la serie reservada para prueba.

    Returns:
        El resultado del entrenamiento, o ``None`` si no hay datos suficientes o
        la variable objetivo no presenta ambas clases.
    """
    if target not in frame.columns:
        return None

    # La temperatura mínima define el objetivo por construcción: incluirla haría
    # el problema trivial y la importancia de las demás variables, ilegible.
    predictors = [
        feature
        for feature in features
        if feature in frame.columns
        and feature not in {target, "temperatura_minima", "deficit_termico", "indice_riesgo_helada"}
    ]
    if not predictors:
        return None

    data = frame.loc[:, [*predictors, target, "fecha"]].dropna().sort_values("fecha")
    if len(data) < 500 or data[target].nunique() < 2:
        return None

    x = data[predictors].to_numpy(dtype="float64")
    y = data[target].to_numpy(dtype="int8")

    split = int(len(data) * (1 - test_size))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        # Con una partición cronológica esto puede ocurrir si el objetivo es muy
        # raro; en tal caso se recurre a una partición estratificada y se acepta
        # la limitación, que queda documentada en el informe.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)

    permutation = permutation_importance(
        model, x_test, y_test, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    importances = (
        pd.DataFrame(
            {
                "variable": predictors,
                "importancia": permutation.importances_mean,
                "desviacion": permutation.importances_std,
            }
        )
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    importances["etiqueta"] = importances["variable"].map(lambda name: variable_label(name, with_unit=False))

    matrix = confusion_matrix(y_test, predictions)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    precision_points, recall_points, _ = precision_recall_curve(y_test, probabilities)

    return ClassifierResult(
        importances=importances,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        average_precision=float(average_precision_score(y_test, probabilities)),
        confusion=matrix,
        roc_curve_points=(false_positive_rate, true_positive_rate),
        pr_curve_points=(recall_points, precision_points),
        accuracy=float((true_positive + true_negative) / matrix.sum()),
        recall=float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0,
        precision=float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else 0.0,
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        positive_rate=float(y.mean()),
        features=predictors,
    )


def project_provinces(
    province_summary: pd.DataFrame, features: list[str], *, n_components: int = 2
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Proyecta los perfiles provinciales en un espacio de baja dimensión.

    Trece provincias descritas por doce variables no pueden compararse a simple
    vista. La proyección resuelve esa limitación: sitúa cada provincia en un
    plano donde la cercanía significa semejanza agroclimática.

    Args:
        province_summary: Resumen por provincia.
        features: Variables del perfil.
        n_components: Dimensiones de la proyección.

    Returns:
        Tupla ``(dataframe_con_coordenadas, varianza_explicada)``, o ``None`` si
        no hay provincias o variables suficientes.
    """
    available = [feature for feature in features if feature in province_summary.columns]
    if len(province_summary) < 3 or len(available) < 2:
        return None

    data = province_summary.loc[:, available].fillna(province_summary[available].mean())
    matrix = StandardScaler().fit_transform(data.to_numpy(dtype="float64"))

    components = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
    model = PCA(n_components=components, random_state=RANDOM_STATE)
    coordinates = model.fit_transform(matrix)

    result = province_summary.copy()
    for index in range(components):
        result[f"CP{index + 1}"] = coordinates[:, index]
    return result, model.explained_variance_ratio_
