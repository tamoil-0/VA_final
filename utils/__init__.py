"""Capa de utilidades: lógica de dominio independiente de la interfaz.

Ningún módulo de este paquete importa Streamlit ni conoce la existencia del
dashboard. Esa separación permite reutilizar el mismo código desde los scripts
de línea de comandos, desde el cuaderno de análisis y desde las pruebas
automatizadas, sin duplicar una sola función.

Módulos
-------
``data_acquisition``
    Cliente de la API NASA POWER y consolidación del dataset original.
``preprocessing``
    Pipeline auditable de limpieza, imputación y generación de variables.
``io``
    Carga con caché, exportación a CSV/Excel y lectura de metadatos.
``filters``
    Aplicación del estado de filtros global sobre el dataframe.
``metrics``
    Cálculo de indicadores, rankings y agregaciones.
``stats_tools``
    Regresión OLS, Mann-Kendall, pendiente de Sen y correlaciones con
    significancia estadística.
``ml_models``
    PCA, K-Means, clustering jerárquico, Isolation Forest y Random Forest.
``insights``
    Motor de hallazgos y conclusiones redactadas automáticamente.
``formatting``
    Formato de números, fechas y unidades para la interfaz.
"""

__all__ = [
    "data_acquisition",
    "filters",
    "formatting",
    "insights",
    "io",
    "metrics",
    "ml_models",
    "preprocessing",
    "stats_tools",
]
