"""Configuración central del proyecto.

Este módulo concentra:

* Rutas del proyecto (`PROJECT_PATHS`).
* Metadatos institucionales y de autoría (`COURSE`, `AUTHORS`).
* Procedencia y trazabilidad del dataset (`DATA_SOURCE`).
* Parámetros del dominio agroclimático (`PROVINCES`, `FROST_THRESHOLDS`, ...).
* Diccionario de variables (`VARIABLE_DICTIONARY`).
* Preguntas de análisis que estructuran el dashboard (`RESEARCH_QUESTIONS`).

Al mantener estos valores en un único lugar se evita la duplicación de
constantes entre la capa de datos, la capa de componentes y las páginas del
dashboard, y se facilita la auditoría del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    """Contenedor inmutable con las rutas relevantes del proyecto.

    Attributes:
        root: Directorio raíz del repositorio.
        data: Directorio de datasets.
        raw_data: Respuestas crudas de la API, una por punto geográfico.
        original_dataset: Dataset consolidado sin procesar.
        clean_dataset: Dataset analítico final.
        data_dictionary: Diccionario de variables en formato CSV.
        metadata: Metadatos de trazabilidad del proceso ETL.
        assets: Recursos estáticos (logos, capturas).
        styles: Hojas de estilo CSS del dashboard.
        report: Fuentes LaTeX del informe técnico.
        screenshots: Capturas del dashboard usadas en el informe.
    """

    root: Path
    data: Path
    raw_data: Path
    original_dataset: Path
    clean_dataset: Path
    data_dictionary: Path
    metadata: Path
    assets: Path
    logos: Path
    screenshots: Path
    styles: Path
    notebooks: Path
    report: Path
    report_figures: Path
    deliverable_report: Path

    def ensure_directories(self) -> None:
        """Crea los directorios necesarios si aún no existen."""
        for directory in (
            self.data,
            self.raw_data,
            self.assets,
            self.logos,
            self.screenshots,
            self.styles,
            self.notebooks,
            self.report,
            self.report_figures,
            self.deliverable_report,
        ):
            directory.mkdir(parents=True, exist_ok=True)


PROJECT_PATHS: Final[ProjectPaths] = ProjectPaths(
    root=ROOT_DIR,
    data=ROOT_DIR / "data",
    raw_data=ROOT_DIR / "data" / "raw",
    original_dataset=ROOT_DIR / "data" / "datos_originales.csv",
    clean_dataset=ROOT_DIR / "data" / "datos_limpios.csv",
    data_dictionary=ROOT_DIR / "data" / "diccionario_datos.csv",
    metadata=ROOT_DIR / "data" / "metadata.json",
    assets=ROOT_DIR / "assets",
    logos=ROOT_DIR / "assets" / "logos",
    screenshots=ROOT_DIR / "assets" / "capturas",
    styles=ROOT_DIR / "styles",
    notebooks=ROOT_DIR / "notebooks",
    report=ROOT_DIR / "report",
    report_figures=ROOT_DIR / "report" / "figuras",
    deliverable_report=ROOT_DIR / "informe",
)

# ---------------------------------------------------------------------------
# Identificación institucional y académica
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Author:
    """Integrante del equipo de trabajo.

    Attributes:
        full_name: Nombre completo en formato "Apellidos, Nombres".
        display_name: Nombre en orden natural, para el encabezado del dashboard.
        role: Rol principal asumido dentro del proyecto.
        contributions: Aportes concretos, exigidos por la modalidad de trabajo.
    """

    full_name: str
    display_name: str
    role: str
    contributions: tuple[str, ...]


AUTHORS: Final[tuple[Author, ...]] = (
    Author(
        full_name="Aracayo Mamani, Jhon Marco",
        display_name="Jhon Marco Aracayo Mamani",
        role="Arquitectura de software y desarrollo del dashboard",
        contributions=(
            "Diseño de la arquitectura por capas y de los componentes reutilizables.",
            "Implementación del dashboard en Streamlit y del sistema de filtros global.",
            "Integración de la capa de exportación (CSV / Excel) y del motor de insights.",
        ),
    ),
    Author(
        full_name="Canaza Paucara, Juan Diego",
        display_name="Juan Diego Canaza Paucara",
        role="Ingeniería de datos y análisis multivariante",
        contributions=(
            "Adquisición y consolidación de los datos desde la API NASA POWER.",
            "Pipeline de limpieza, imputación y generación de variables derivadas.",
            "Implementación de PCA, K-Means, clustering jerárquico y detección de anomalías.",
        ),
    ),
    Author(
        full_name="Luque Pacheco, Angie Tatiana",
        display_name="Angie Tatiana Luque Pacheco",
        role="Visualización de datos, análisis estadístico y redacción técnica",
        contributions=(
            "Diseño de la identidad visual, la paleta pastel y la jerarquía gráfica.",
            "Construcción y justificación analítica de las visualizaciones.",
            "Análisis de tendencias, correlaciones y redacción del informe técnico.",
        ),
    ),
)

COURSE: Final[dict[str, str]] = {
    "university": "Universidad Nacional del Altiplano",
    "university_short": "UNAP",
    "faculty": "Facultad de Ingeniería Mecánica Eléctrica, Electrónica y Sistemas",
    "school": "Escuela Profesional de Ingeniería de Sistemas",
    "course": "Visualización de Datos",
    "course_code": "SIS328",
    "semester": "2026-I",
    "cycle": "X",
    "professor": "Ing. Edwin Edgar Mestas Yucra",
    "city": "Puno – Perú",
    "year": "2026",
    "project_title": "Riesgo agroclimático por heladas en la región Puno",
    "project_subtitle": (
        "Dashboard interactivo para el análisis y la visualización de datos "
        "orientado a la toma de decisiones"
    ),
    "repository_url": "https://github.com/jhonaracayo/dashboard-heladas-puno",
    "dashboard_url": "https://dashboard-heladas-puno.streamlit.app",
}

PROBLEM_STATEMENT: Final[str] = (
    "Las heladas meteorológicas constituyen el principal peligro agroclimático del "
    "altiplano peruano: reducen el rendimiento de cultivos altoandinos, comprometen la "
    "seguridad alimentaria de las familias productoras y afectan la sanidad del ganado. "
    "Pese a la existencia de registros climáticos diarios de acceso abierto, esta "
    "información permanece dispersa, en formatos poco amigables y sin instrumentos de "
    "exploración que permitan a las autoridades regionales identificar cuándo, dónde y "
    "con qué intensidad se concentra el riesgo. Este dashboard transforma diez años de "
    "observaciones agrometeorológicas de las trece provincias de Puno en indicadores, "
    "visualizaciones y hallazgos accionables para la gestión del riesgo agrícola."
)

# ---------------------------------------------------------------------------
# Procedencia del dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSource:
    """Metadatos de procedencia del conjunto de datos.

    Attributes:
        name: Nombre corto de la fuente.
        organization: Institución responsable de la producción del dato.
        endpoint: URL del servicio consultado.
        portal: Portal de documentación pública.
        license_note: Condiciones de uso declaradas por el proveedor.
        citation_apa: Referencia lista para el informe, en estilo APA 7.
        model: Modelo o producto de origen de las observaciones.
        spatial_resolution: Resolución nominal de la malla.
    """

    name: str
    organization: str
    endpoint: str
    portal: str
    license_note: str
    citation_apa: str
    model: str
    spatial_resolution: str


#: **Fuente primaria.** Alimenta la totalidad del dashboard. Se eligió ERA5-Land
#: por su resolución espacial de ≈ 9 km: es la mínima que permite diferenciar
#: entre capitales provinciales vecinas del altiplano. Con mallas más gruesas,
#: varias capitales caen en la misma celda y reciben series idénticas, lo que
#: invalidaría cualquier ranking o agrupamiento provincial.
DATA_SOURCE: Final[DataSource] = DataSource(
    name="ERA5-Land / ERA5 (Copernicus Climate Change Service) vía Open-Meteo",
    organization=(
        "European Centre for Medium-Range Weather Forecasts (ECMWF) — Copernicus "
        "Climate Change Service (C3S); acceso mediante Open-Meteo.com"
    ),
    endpoint="https://archive-api.open-meteo.com/v1/archive",
    portal="https://open-meteo.com/en/docs/historical-weather-api",
    license_note=(
        "Datos de Copernicus bajo licencia de uso abierto; el servicio de acceso "
        "Open-Meteo se distribuye bajo CC BY 4.0. Se permite el uso académico con "
        "atribución a ambas partes."
    ),
    citation_apa=(
        "Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., Albergel, C., Arduini, G., "
        "Balsamo, G., Boussetta, S., Choulga, M., Harrigan, S., Hersbach, H., Martens, B., "
        "Miralles, D. G., Piles, M., Rodríguez-Fernández, N. J., Zsoter, E., Buontempo, C., "
        "& Thépaut, J.-N. (2021). ERA5-Land: A state-of-the-art global reanalysis dataset "
        "for land applications. Earth System Science Data, 13(9), 4349–4383. "
        "https://doi.org/10.5194/essd-13-4349-2021"
    ),
    model="Reanálisis ERA5-Land / ERA5 (configuración «era5_seamless»)",
    spatial_resolution="0.1° ≈ 9 km (ERA5-Land); 0.25° ≈ 28 km en las variables servidas por ERA5",
)

#: **Fuente secundaria.** No alimenta el dashboard: se emplea exclusivamente
#: para la validación cruzada del régimen de heladas contra un reanálisis
#: independiente, producido por otra institución y con otro modelo de base.
#: La coincidencia de ambas fuentes es la evidencia de que la señal detectada no
#: es un artefacto de un producto concreto.
DATA_SOURCE_SECONDARY: Final[DataSource] = DataSource(
    name="NASA POWER — Prediction Of Worldwide Energy Resources",
    organization="NASA Langley Research Center (LaRC), Applied Sciences Program",
    endpoint="https://power.larc.nasa.gov/api/temporal/daily/point",
    portal="https://power.larc.nasa.gov/",
    license_note=(
        "Datos de dominio público. NASA POWER autoriza expresamente su uso, "
        "redistribución y publicación citando la fuente."
    ),
    citation_apa=(
        "NASA Langley Research Center. (2025). NASA Prediction Of Worldwide Energy "
        "Resources (POWER) Project: Daily agroclimatology data (Version 2.9.6) "
        "[Conjunto de datos]. https://power.larc.nasa.gov/"
    ),
    model="Reanálisis MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, v2)",
    spatial_resolution="0.5° × 0.625° (≈ 55 km × 65 km)",
)


@dataclass(frozen=True)
class StudyPeriod:
    """Ventana temporal del estudio.

    Attributes:
        start: Primer día incluido en la descarga.
        end: Último día incluido en la descarga.
    """

    start: date
    end: date

    @property
    def label(self) -> str:
        """Devuelve la etiqueta legible del periodo (p. ej. ``2015–2024``)."""
        return f"{self.start.year}–{self.end.year}"

    @property
    def api_start(self) -> str:
        """Fecha inicial en el formato ``YYYYMMDD`` que exige la API."""
        return self.start.strftime("%Y%m%d")

    @property
    def api_end(self) -> str:
        """Fecha final en el formato ``YYYYMMDD`` que exige la API."""
        return self.end.strftime("%Y%m%d")

    @property
    def n_years(self) -> int:
        """Número de años calendario completos cubiertos."""
        return self.end.year - self.start.year + 1


STUDY_PERIOD: Final[StudyPeriod] = StudyPeriod(start=date(2015, 1, 1), end=date(2024, 12, 31))

# ---------------------------------------------------------------------------
# Dominio geográfico: las 13 provincias de la región Puno
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Province:
    """Provincia de la región Puno representada por su capital.

    Attributes:
        name: Nombre de la provincia.
        capital: Capital provincial, punto de consulta de la serie climática.
        latitude: Latitud decimal de la capital (grados, sur negativo).
        longitude: Longitud decimal de la capital (grados, oeste negativo).
        reference_altitude: Altitud referencial de la capital en msnm.
        basin: Vertiente hidrográfica a la que drena el territorio.
        agroecological_zone: Zona agroecológica predominante.
    """

    name: str
    capital: str
    latitude: float
    longitude: float
    reference_altitude: int
    basin: str
    agroecological_zone: str


PROVINCES: Final[tuple[Province, ...]] = (
    Province("Puno", "Puno", -15.8402, -70.0219, 3827, "Titicaca", "Altiplano circunlacustre"),
    Province("Azángaro", "Azángaro", -14.9114, -70.1969, 3859, "Titicaca", "Altiplano intermedio"),
    Province("Carabaya", "Macusani", -14.0703, -70.4297, 4315, "Amazonas", "Puna alta y cordillera"),
    Province("Chucuito", "Juli", -16.2131, -69.4589, 3869, "Titicaca", "Altiplano circunlacustre"),
    Province("El Collao", "Ilave", -16.0872, -69.6664, 3850, "Titicaca", "Altiplano circunlacustre"),
    Province("Huancané", "Huancané", -15.2036, -69.7594, 3841, "Titicaca", "Altiplano circunlacustre"),
    Province("Lampa", "Lampa", -15.3617, -70.3667, 3892, "Titicaca", "Altiplano intermedio"),
    Province("Melgar", "Ayaviri", -14.8797, -70.5897, 3907, "Titicaca", "Altiplano intermedio"),
    Province("Moho", "Moho", -15.3506, -69.4906, 3890, "Titicaca", "Altiplano circunlacustre"),
    Province(
        "San Antonio de Putina", "Putina", -14.9139, -69.8722, 3878, "Titicaca", "Altiplano intermedio"
    ),
    Province("San Román", "Juliaca", -15.4997, -70.1333, 3824, "Titicaca", "Altiplano intermedio"),
    Province("Sandia", "Sandia", -14.3222, -69.4625, 2170, "Amazonas", "Valles interandinos y ceja de selva"),
    Province("Yunguyo", "Yunguyo", -16.2447, -69.0906, 3860, "Titicaca", "Altiplano circunlacustre"),
)

PROVINCE_NAMES: Final[tuple[str, ...]] = tuple(province.name for province in PROVINCES)

#: Pisos ecológicos según la clasificación altitudinal de Pulgar Vidal (1981),
#: empleada por el MINAM para la caracterización de los Andes peruanos.
ECOLOGICAL_TIERS: Final[tuple[tuple[str, float, float], ...]] = (
    ("Selva alta / Yunga", -float("inf"), 2300.0),
    ("Quechua", 2300.0, 3500.0),
    ("Suni / Altiplano", 3500.0, 4000.0),
    ("Puna", 4000.0, 4800.0),
    ("Janca / Cordillera", 4800.0, float("inf")),
)

ECOLOGICAL_TIER_ORDER: Final[tuple[str, ...]] = tuple(name for name, _, _ in ECOLOGICAL_TIERS)

# ---------------------------------------------------------------------------
# Parámetros del dominio agroclimático
# ---------------------------------------------------------------------------

#: Umbrales de temperatura mínima (°C) usados operativamente por SENAMHI.
#: ``meteorologica``: temperatura del aire a 2 m igual o inferior a 0 °C.
#: ``agronomica``: umbral de daño fisiológico en cultivos altoandinos sensibles.
FROST_THRESHOLDS: Final[dict[str, float]] = {
    "meteorologica": 0.0,
    "agronomica": 3.0,
}

FROST_THRESHOLD_LABELS: Final[dict[str, str]] = {
    "meteorologica": "Helada meteorológica (Tmín ≤ 0 °C)",
    "agronomica": "Helada agronómica (Tmín ≤ 3 °C)",
}

#: Cortes de intensidad de helada en °C. Cada tupla es (etiqueta, límite inferior
#: exclusivo, límite superior inclusivo) sobre la temperatura mínima diaria.
FROST_INTENSITY_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("Sin helada", 0.0, float("inf")),
    ("Ligera", -2.0, 0.0),
    ("Moderada", -4.0, -2.0),
    ("Severa", -6.0, -4.0),
    ("Extrema", -float("inf"), -6.0),
)

FROST_INTENSITY_ORDER: Final[tuple[str, ...]] = (
    "Sin helada",
    "Ligera",
    "Moderada",
    "Severa",
    "Extrema",
)

#: Estacionalidad del altiplano peruano. La campaña agrícola nacional se
#: computa de agosto a julio del año siguiente (MIDAGRI).
SEASON_BY_MONTH: Final[dict[int, str]] = {
    1: "Lluviosa",
    2: "Lluviosa",
    3: "Lluviosa",
    4: "Transición",
    5: "Seca",
    6: "Seca",
    7: "Seca",
    8: "Seca",
    9: "Seca",
    10: "Transición",
    11: "Lluviosa",
    12: "Lluviosa",
}

SEASON_ORDER: Final[tuple[str, ...]] = ("Lluviosa", "Transición", "Seca")

AGRICULTURAL_CAMPAIGN_START_MONTH: Final[int] = 8

MONTH_LABELS: Final[dict[int, str]] = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

MONTH_ABBREV: Final[dict[int, str]] = {
    1: "Ene",
    2: "Feb",
    3: "Mar",
    4: "Abr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Set",
    10: "Oct",
    11: "Nov",
    12: "Dic",
}

RISK_LEVEL_ORDER: Final[tuple[str, ...]] = ("Bajo", "Moderado", "Alto", "Muy alto", "Crítico")

# ---------------------------------------------------------------------------
# Variables del modelo de datos
# ---------------------------------------------------------------------------

#: Variables diarias solicitadas a la fuente primaria (Open-Meteo / ERA5-Land).
#: La clave es el identificador nativo del servicio y el valor, el nombre
#: semántico en español que adopta la columna en el dataset limpio.
ERA5_DAILY_VARIABLES: Final[dict[str, str]] = {
    "temperature_2m_mean": "temperatura_media",
    "temperature_2m_max": "temperatura_maxima",
    "temperature_2m_min": "temperatura_minima",
    "dew_point_2m_mean": "punto_rocio",
    "relative_humidity_2m_mean": "humedad_relativa",
    "precipitation_sum": "precipitacion",
    "rain_sum": "lluvia",
    "snowfall_sum": "nevada",
    "precipitation_hours": "horas_precipitacion",
    "shortwave_radiation_sum": "radiacion_solar",
    "wind_speed_10m_mean": "velocidad_viento",
    "wind_speed_10m_max": "racha_viento_maxima",
    "surface_pressure_mean": "presion_superficie",
    "cloud_cover_mean": "nubosidad",
    "et0_fao_evapotranspiration": "evapotranspiracion",
    "vapour_pressure_deficit_max": "deficit_presion_vapor_maximo",
    "soil_temperature_0_to_7cm_mean": "temperatura_suelo",
    "soil_moisture_0_to_7cm_mean": "humedad_suelo",
    "sunshine_duration": "horas_sol",
}

#: Modelo solicitado explícitamente al servicio. Se fija en lugar de aceptar la
#: selección automática para que el dataset sea reproducible y atribuible a una
#: familia de reanálisis única y citable.
ERA5_MODEL: Final[str] = "era5_seamless"

#: Zona horaria de agregación diaria. Es determinante: la temperatura mínima de
#: un día y, por tanto, la ocurrencia de una helada, dependen de dónde se corta
#: la jornada. Se emplea la hora local peruana, no UTC.
LOCAL_TIMEZONE: Final[str] = "America/Lima"

#: Conversiones de unidad aplicadas en la transformación. El servicio entrega
#: viento en km/h, presión en hPa e insolación en segundos; el dataset analítico
#: adopta las unidades convencionales en agrometeorología.
UNIT_CONVERSIONS: Final[dict[str, tuple[float, str, str]]] = {
    "velocidad_viento": (1 / 3.6, "km/h", "m/s"),
    "racha_viento_maxima": (1 / 3.6, "km/h", "m/s"),
    "presion_superficie": (0.1, "hPa", "kPa"),
    "horas_sol": (1 / 3600, "s", "h"),
}

#: Parámetros solicitados a la fuente secundaria (NASA POWER, comunidad ``AG``),
#: empleados sólo en la validación cruzada.
MERRA2_PARAMETERS: Final[dict[str, str]] = {
    "T2M": "temperatura_media_merra2",
    "T2M_MAX": "temperatura_maxima_merra2",
    "T2M_MIN": "temperatura_minima_merra2",
    "PRECTOTCORR": "precipitacion_merra2",
}

#: Valor centinela que la API de NASA POWER emplea para datos no disponibles.
API_FILL_VALUE: Final[float] = -999.0


@dataclass(frozen=True)
class VariableSpec:
    """Especificación semántica de una variable del dataset limpio.

    Attributes:
        name: Nombre de la columna.
        label: Etiqueta legible para la interfaz.
        unit: Unidad de medida (vacía si es adimensional).
        dtype: Tipo lógico: ``numerica``, ``categorica``, ``temporal``,
            ``geografica`` o ``booleana``.
        origin: ``original`` si proviene de la fuente, ``derivada`` si se
            construyó en el pipeline de transformación.
        description: Definición operativa de la variable.
    """

    name: str
    label: str
    unit: str
    dtype: str
    origin: str
    description: str


VARIABLE_DICTIONARY: Final[tuple[VariableSpec, ...]] = (
    VariableSpec("fecha", "Fecha", "", "temporal", "original", "Día de observación (resolución diaria)."),
    VariableSpec("provincia", "Provincia", "", "categorica", "original", "Provincia de la región Puno."),
    VariableSpec("capital", "Capital provincial", "", "categorica", "original", "Capital que define el punto de consulta."),
    VariableSpec("latitud", "Latitud", "°", "geografica", "original", "Latitud decimal del punto de malla."),
    VariableSpec("longitud", "Longitud", "°", "geografica", "original", "Longitud decimal del punto de malla."),
    VariableSpec("altitud", "Altitud", "msnm", "numerica", "original", "Elevación del punto reportada por la API."),
    VariableSpec("cuenca", "Vertiente hidrográfica", "", "categorica", "derivada", "Titicaca (endorreica) o Amazonas."),
    VariableSpec("zona_agroecologica", "Zona agroecológica", "", "categorica", "derivada", "Caracterización agroecológica provincial."),
    VariableSpec("piso_ecologico", "Piso ecológico", "", "categorica", "derivada", "Piso altitudinal según Pulgar Vidal (1981)."),
    VariableSpec("temperatura_media", "Temperatura media", "°C", "numerica", "original", "Temperatura del aire a 2 m, promedio diario."),
    VariableSpec("temperatura_maxima", "Temperatura máxima", "°C", "numerica", "original", "Temperatura máxima diaria del aire a 2 m."),
    VariableSpec("temperatura_minima", "Temperatura mínima", "°C", "numerica", "original", "Temperatura mínima diaria del aire a 2 m; variable central del estudio."),
    VariableSpec("punto_rocio", "Punto de rocío", "°C", "numerica", "original", "Temperatura de rocío media a 2 m."),
    VariableSpec("humedad_relativa", "Humedad relativa", "%", "numerica", "original", "Humedad relativa media diaria a 2 m."),
    VariableSpec("precipitacion", "Precipitación", "mm", "numerica", "original", "Precipitación total diaria (lluvia y equivalente en agua de nieve)."),
    VariableSpec("lluvia", "Lluvia", "mm", "numerica", "original", "Fracción líquida de la precipitación diaria."),
    VariableSpec("nevada", "Nevada", "cm", "numerica", "original", "Espesor de nieve acumulada en el día."),
    VariableSpec("horas_precipitacion", "Horas con precipitación", "h", "numerica", "original", "Número de horas del día con precipitación registrada."),
    VariableSpec("radiacion_solar", "Radiación solar", "MJ/m²", "numerica", "original", "Irradiancia global horizontal acumulada en el día."),
    VariableSpec("horas_sol", "Horas de sol", "h", "numerica", "original", "Duración de la insolación directa."),
    VariableSpec("nubosidad", "Nubosidad", "%", "numerica", "original", "Cobertura nubosa media diaria; determinante de la helada de radiación."),
    VariableSpec("velocidad_viento", "Velocidad del viento", "m/s", "numerica", "original", "Velocidad media del viento a 10 m."),
    VariableSpec("racha_viento_maxima", "Racha máxima de viento", "m/s", "numerica", "original", "Velocidad máxima del viento a 10 m."),
    VariableSpec("presion_superficie", "Presión superficial", "kPa", "numerica", "original", "Presión atmosférica media en superficie."),
    VariableSpec("evapotranspiracion", "Evapotranspiración de referencia", "mm", "numerica", "original", "ET₀ diaria calculada con el método FAO-56 Penman-Monteith."),
    VariableSpec("deficit_presion_vapor_maximo", "Déficit de presión de vapor máximo", "kPa", "numerica", "original", "Valor máximo diario del déficit de presión de vapor."),
    VariableSpec("temperatura_suelo", "Temperatura del suelo", "°C", "numerica", "original", "Temperatura media del suelo entre 0 y 7 cm de profundidad."),
    VariableSpec("humedad_suelo", "Humedad del suelo", "m³/m³", "numerica", "original", "Contenido volumétrico de agua del suelo entre 0 y 7 cm."),
    VariableSpec("temperatura_minima_merra2", "Temperatura mínima (MERRA-2)", "°C", "numerica", "original", "Temperatura mínima de la fuente secundaria, para validación cruzada."),
    VariableSpec("helada_merra2", "Helada según MERRA-2", "", "booleana", "derivada", "Ocurrencia de helada según la fuente secundaria independiente."),
    VariableSpec("anio", "Año", "", "temporal", "derivada", "Año calendario de la observación."),
    VariableSpec("mes", "Mes", "", "temporal", "derivada", "Número de mes (1–12)."),
    VariableSpec("mes_nombre", "Nombre del mes", "", "categorica", "derivada", "Mes en texto, ordenado cronológicamente."),
    VariableSpec("trimestre", "Trimestre", "", "categorica", "derivada", "Trimestre calendario (T1–T4)."),
    VariableSpec("dia_anio", "Día del año", "", "temporal", "derivada", "Día juliano (1–366)."),
    VariableSpec("temporada", "Temporada", "", "categorica", "derivada", "Lluviosa, Transición o Seca."),
    VariableSpec("campania_agricola", "Campaña agrícola", "", "categorica", "derivada", "Campaña MIDAGRI de agosto a julio."),
    VariableSpec("oscilacion_termica", "Oscilación térmica", "°C", "numerica", "derivada", "Diferencia entre temperatura máxima y mínima."),
    VariableSpec("helada", "Helada meteorológica", "", "booleana", "derivada", "Verdadero si la temperatura mínima ≤ 0 °C."),
    VariableSpec("helada_agronomica", "Helada agronómica", "", "booleana", "derivada", "Verdadero si la temperatura mínima ≤ 3 °C."),
    VariableSpec("intensidad_helada", "Intensidad de helada", "", "categorica", "derivada", "Clasificación ordinal de severidad del evento."),
    VariableSpec("deficit_termico", "Déficit térmico", "°C", "numerica", "derivada", "Grados por debajo del umbral agronómico (0 si no aplica)."),
    VariableSpec("dia_lluvia", "Día con lluvia", "", "booleana", "derivada", "Verdadero si la precipitación ≥ 1 mm."),
    VariableSpec("precipitacion_sospechosa", "Precipitación implausible", "", "booleana", "derivada", "Marca de control de calidad: acumulado diario > 100 mm, atribuible al reanálisis."),
    VariableSpec("nivel_riesgo", "Nivel de riesgo", "", "categorica", "derivada", "Clasificación ordinal del índice de riesgo en cinco niveles."),
    VariableSpec("deficit_presion_vapor", "Déficit de presión de vapor", "kPa", "numerica", "derivada", "VPD calculado con la ecuación de Tetens."),
    VariableSpec("indice_riesgo_helada", "Índice de riesgo de helada", "0–100", "numerica", "derivada", "Índice compuesto de riesgo agroclimático diario."),
)

#: Variables numéricas ofrecidas al usuario en los selectores de «variable de
#: análisis». Se excluyen deliberadamente las variables redundantes por
#: construcción (``lluvia`` respecto de ``precipitacion``) y las de validación
#: cruzada, que no son objeto de análisis sino de control.
NUMERIC_ANALYSIS_VARIABLES: Final[tuple[str, ...]] = (
    "temperatura_minima",
    "temperatura_media",
    "temperatura_maxima",
    "oscilacion_termica",
    "temperatura_suelo",
    "precipitacion",
    "humedad_relativa",
    "punto_rocio",
    "deficit_presion_vapor",
    "radiacion_solar",
    "horas_sol",
    "nubosidad",
    "velocidad_viento",
    "evapotranspiracion",
    "humedad_suelo",
    "presion_superficie",
    "indice_riesgo_helada",
    "altitud",
)

#: Variables empleadas en el análisis multivariante (PCA, agrupamiento,
#: detección de anomalías). La selección evita la multicolinealidad extrema por
#: identidad algebraica: se incluye ``oscilacion_termica`` pero no simultáneamente
#: sus dos términos generadores, y se prefiere ``punto_rocio`` frente a
#: ``humedad_relativa`` cuando ambas saturarían el mismo eje.
MULTIVARIATE_FEATURES: Final[tuple[str, ...]] = (
    "temperatura_minima",
    "oscilacion_termica",
    "temperatura_suelo",
    "precipitacion",
    "humedad_relativa",
    "punto_rocio",
    "radiacion_solar",
    "nubosidad",
    "velocidad_viento",
    "evapotranspiracion",
    "humedad_suelo",
    "altitud",
)

# ---------------------------------------------------------------------------
# Preguntas de análisis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchQuestion:
    """Pregunta analítica que el dashboard debe responder.

    Attributes:
        code: Identificador corto (``P1``…``P6``).
        question: Enunciado de la pregunta.
        page: Página del dashboard donde se responde.
        method: Técnica o visualización principal utilizada.
    """

    code: str
    question: str
    page: str
    method: str


RESEARCH_QUESTIONS: Final[tuple[ResearchQuestion, ...]] = (
    ResearchQuestion(
        code="P1",
        question=(
            "¿Cuál es la situación general del riesgo por heladas en la región Puno "
            "durante el periodo 2015–2024?"
        ),
        page="Vista general",
        method="Indicadores clave, resumen ejecutivo y perfil climático anual",
    ),
    ResearchQuestion(
        code="P2",
        question=(
            "¿Cómo se distribuye la ocurrencia e intensidad de las heladas entre las "
            "provincias y a lo largo del ciclo anual?"
        ),
        page="Análisis descriptivo",
        method="Gráfico de barras, histograma, diagrama de caja y mapa de calor estacional",
    ),
    ResearchQuestion(
        code="P3",
        question=(
            "¿Cómo ha evolucionado la frecuencia e intensidad de las heladas y existe "
            "una tendencia estadísticamente significativa?"
        ),
        page="Análisis descriptivo",
        method="Serie temporal, regresión OLS, prueba de Mann-Kendall y pendiente de Sen",
    ),
    ResearchQuestion(
        code="P4",
        question="¿Qué provincias concentran el mayor y el menor riesgo agroclimático?",
        page="Análisis descriptivo / Geográfico",
        method="Ranking ordenado, mapa coroplético de puntos y sistema de alertas",
    ),
    ResearchQuestion(
        code="P5",
        question=(
            "¿Qué relación existe entre la altitud, la humedad, la radiación y la "
            "precipitación con la temperatura mínima diaria?"
        ),
        page="Análisis multidimensional",
        method="Dispersión con regresión, matriz de correlación Pearson/Spearman y VIF",
    ),
    ResearchQuestion(
        code="P6",
        question=(
            "¿Es posible identificar zonas agroclimáticas homogéneas, factores latentes "
            "y días atípicos mediante técnicas multivariantes?"
        ),
        page="Análisis multidimensional",
        method="PCA, K-Means, clustering jerárquico, Isolation Forest y Random Forest",
    ),
)

# ---------------------------------------------------------------------------
# Parámetros de la aplicación
# ---------------------------------------------------------------------------

APP_CONFIG: Final[dict[str, object]] = {
    "page_icon": "❄️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "Get help": COURSE["repository_url"],
        "Report a bug": f"{COURSE['repository_url']}/issues",
        "About": (
            f"**{COURSE['project_title']}**\n\n"
            f"{COURSE['course']} ({COURSE['course_code']}) — {COURSE['school']}, "
            f"{COURSE['university']}.\n\n"
            f"Datos: {DATA_SOURCE.name}."
        ),
    },
}

#: Tamaño máximo de muestra usado en gráficos de dispersión densos, para
#: preservar la fluidez de la interfaz sin sesgar la lectura visual.
SCATTER_SAMPLE_SIZE: Final[int] = 6_000

#: Semilla global: garantiza que PCA, K-Means, Isolation Forest y cualquier
#: submuestreo sean exactamente reproducibles entre ejecuciones.
RANDOM_STATE: Final[int] = 42


def get_province(name: str) -> Province:
    """Devuelve la provincia cuyo nombre coincide con ``name``.

    Args:
        name: Nombre de la provincia a buscar.

    Returns:
        La instancia :class:`Province` correspondiente.

    Raises:
        KeyError: Si el nombre no corresponde a ninguna provincia registrada.
    """
    for province in PROVINCES:
        if province.name == name:
            return province
    raise KeyError(f"Provincia no registrada en la configuración: {name!r}")


def classify_ecological_tier(altitude: float) -> str:
    """Clasifica una altitud en su piso ecológico altitudinal.

    Args:
        altitude: Elevación en metros sobre el nivel del mar.

    Returns:
        Nombre del piso ecológico correspondiente.
    """
    for tier, lower, upper in ECOLOGICAL_TIERS:
        if lower <= altitude < upper:
            return tier
    return ECOLOGICAL_TIER_ORDER[-1]
