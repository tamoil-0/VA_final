"""Capa de componentes reutilizables de la interfaz.

Cada componente encapsula un patrón visual completo —encabezado, tarjeta de
indicador, bloque de interpretación, panel de descargas— de modo que las páginas
se limiten a orquestar componentes y no contengan marcado HTML ni decisiones de
estilo. Esa separación es lo que garantiza que las cinco páginas del dashboard
sean visualmente coherentes sin repetir una sola línea de presentación.

Módulos
-------
``layout``
    Configuración de página, inyección de estilos, encabezado y pie.
``sidebar``
    Panel de filtros global y su estado compartido entre páginas.
``kpi``
    Tarjetas de indicador con variación y ayuda contextual.
``cards``
    Tarjetas de sección, bloques de interpretación y hallazgos.
``charts``
    Constructores de todas las visualizaciones del proyecto.
``downloads``
    Panel de exportación en CSV, Excel e imagen.
"""

from components import cards, charts, downloads, kpi, layout, sidebar

__all__ = ["cards", "charts", "downloads", "kpi", "layout", "sidebar"]
