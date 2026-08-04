"""Sistema de diseño e identidad visual del dashboard.

Arquitectura de color en dos niveles
------------------------------------
La identidad visual del proyecto es **pastel** (naranja pastel como color
principal y azul pastel como secundario). Sin embargo, un color pastel puro
—claridad alta y croma bajo— no puede sostener por sí solo la *identidad de
serie* en un gráfico: dos pastel contiguos se vuelven indistinguibles para
personas con deficiencia en la visión del color. Para resolver esa tensión sin
sacrificar ninguno de los dos objetivos, el sistema separa dos niveles:

1. **Nivel de superficie (pastel).** Fondos, tarjetas, degradados, insignias,
   barras de KPI y todo el cromo de la interfaz. Es la mayor parte del área
   visible, y es la que comunica la elegancia pastel solicitada.
2. **Nivel de tinta de datos (pastel legible).** Colores de las marcas
   gráficas. Se derivan de las mismas familias de tono (naranja, azul, …) pero
   con los pasos de claridad y croma necesarios para superar las pruebas de
   accesibilidad.

Verificación cuantitativa
-------------------------
Ninguna paleta de este módulo fue elegida a ojo. Todas se obtuvieron por
optimización con restricciones en el espacio OKLCH —minimizando la saturación
(objetivo estético) sujeto a superar todas las puertas de accesibilidad
(objetivo funcional)— y se validaron con el verificador de paletas del sistema
de diseño. Los resultados de esa validación se documentan junto a cada
constante, con las métricas expresadas como distancia euclídea ΔE en OKLab×100
bajo simulación de protanopía y deuteranopía (Machado-Oliveira-Fernandes, 2009,
severidad 1.0).

Reglas de uso obligatorias
--------------------------
* ``CATEGORICAL_FILL`` sólo en formas donde únicamente los vecinos se tocan
  (barras, líneas, áreas apiladas). Como sus contrastes están por debajo de
  3:1, todo gráfico que la use **debe** ofrecer canal de relevo: etiquetas
  visibles directas y/o la vista de tabla asociada.
* ``CATEGORICAL_ALL_PAIRS`` en dispersión, burbujas, mapas y pequeños
  múltiplos, donde cualquier par de marcas puede quedar adyacente. Está
  limitada a **cuatro** series por diseño; con más categorías se agrupa en
  «Otros» o se recurre a facetas.
* ``CATEGORICAL_INK`` para trazos finos (líneas de 2 px, contornos, bordes),
  donde el contraste mínimo de 3:1 es indispensable.
* ``SEQUENTIAL_*`` para magnitud continua, ``ORDINAL_FROST`` para la severidad
  ordinal de las heladas y ``DIVERGING_THERMAL`` para anomalías con signo.
* ``STATUS`` tiene semántica reservada y nunca se reutiliza como color de
  serie; siempre acompaña a un icono y a una etiqueta de texto.
"""

from __future__ import annotations

from typing import Final

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Superficies y neutros (grises azulados, según la identidad solicitada)
# ---------------------------------------------------------------------------

NEUTRAL: Final[dict[str, str]] = {
    "0": "#FFFFFF",
    "25": "#FBFCFD",
    "50": "#F7F8FA",
    "100": "#F1F3F7",
    "200": "#E6E9EF",
    "300": "#D5DBE5",
    "400": "#AEB7C4",
    "500": "#7C8798",
    "600": "#5A6373",
    "700": "#3A414D",
    "800": "#232935",
    "900": "#111827",
}

#: Escala de naranja (color principal). Derivada como rampa secuencial de tono
#: único y claridad monótona: ΔL adyacente ≥ 0.06 en OKLCH, verificado.
ORANGE: Final[dict[str, str]] = {
    "50": "#FEECE4",
    "100": "#FBC8B1",
    "200": "#F8A37B",
    "300": "#DB8963",
    "400": "#BB7350",
    "500": "#9A5E41",
    "600": "#794B35",
}

#: Escala de azul (color secundario), construida con el mismo procedimiento.
BLUE: Final[dict[str, str]] = {
    "50": "#E7F1FE",
    "100": "#BAD8FB",
    "200": "#8CBFF9",
    "300": "#6BA5E5",
    "400": "#578BC5",
    "500": "#4772A2",
    "600": "#3A5A7F",
}

# ---------------------------------------------------------------------------
# Paletas categóricas validadas
# ---------------------------------------------------------------------------

#: Relleno pastel de ocho ranuras, orden fijo. Validación (superficie #FFFFFF,
#: modo claro, lista de pares adyacentes):
#:   banda de claridad PASS · piso de croma PASS
#:   separación CVD PASS — peor par adyacente #D07E7A↔#62C6AE ΔE 8.5 (deutan)
#:   piso de visión normal PASS — peor par #62C6AE↔#ADA7EE ΔE 16.7
#:   contraste RELEVO — seis pasos por debajo de 3:1 ⇒ obliga etiquetas
#:   visibles o vista de tabla (ambas se entregan en el dashboard).
CATEGORICAL_FILL: Final[tuple[str, ...]] = (
    "#E79D7A",  # 1 naranja pastel  (principal)
    "#81B5F0",  # 2 azul pastel     (secundario)
    "#D2AA63",  # 3 ámbar
    "#B16989",  # 4 ciruela
    "#92C486",  # 5 verde
    "#ADA7EE",  # 6 violeta
    "#62C6AE",  # 7 aqua
    "#D07E7A",  # 8 rojo suave
)

#: Variante de tinta: mismas familias de tono, pasos oscurecidos hasta superar
#: 3:1 sobre blanco. Se usa en trazos finos, contornos y marcadores pequeños.
#: Validación (pares adyacentes): banda PASS · croma PASS ·
#: CVD PASS (peor #4C7A3F↔#A76080 ΔE 8.3 deutan) ·
#: visión normal PASS (peor #3CA38C↔#B58E46 ΔE 15.0) · contraste PASS (todos ≥3:1).
CATEGORICAL_INK: Final[tuple[str, ...]] = (
    "#BB7554",  # 1 naranja
    "#6396CF",  # 2 azul
    "#A76080",  # 3 ciruela
    "#4C7A3F",  # 4 verde
    "#B58E46",  # 5 ámbar
    "#3CA38C",  # 6 aqua
    "#918BD1",  # 7 violeta
    "#D07E7A",  # 8 rojo suave
)

#: Paleta para formas donde **cualquier** par de marcas puede quedar contiguo
#: (dispersión, burbujas, mapas, pequeños múltiplos). Tope estructural: cuatro
#: series. Validación con lista de pares completa:
#:   banda PASS · croma PASS · CVD PASS (peor #6D9F60↔#CB508D ΔE 8.0 deutan) ·
#:   visión normal PASS (peor #CB508D↔#9E5A39 ΔE 15.3) · contraste PASS (≥3:1).
CATEGORICAL_ALL_PAIRS: Final[tuple[str, ...]] = (
    "#9E5A39",  # 1 naranja profundo
    "#6396CF",  # 2 azul
    "#CB508D",  # 3 magenta
    "#6D9F60",  # 4 verde
)

#: Número máximo de series admitido en formas de tipo «todos los pares».
MAX_ALL_PAIRS_SERIES: Final[int] = len(CATEGORICAL_ALL_PAIRS)

# ---------------------------------------------------------------------------
# Rampas de magnitud, orden y polaridad
# ---------------------------------------------------------------------------

#: Rampa secuencial principal (azul), claro→oscuro, para magnitud continua:
#: mapas de calor y mapas coropléticos. Monotonía de claridad verificada
#: (ΔL adyacente ≥ 0.06, dispersión de tono 3°). El extremo claro cae por
#: debajo de 2:1 de forma deliberada: en codificación secuencial representa
#: «cerca de cero» y debe retroceder hacia la superficie.
SEQUENTIAL_BLUE: Final[tuple[str, ...]] = (
    "#E7F1FE",
    "#BAD8FB",
    "#8CBFF9",
    "#6BA5E5",
    "#578BC5",
    "#4772A2",
    "#3A5A7F",
)

#: Rampa secuencial secundaria (naranja), para cuando coexisten dos contextos
#: de magnitud en la misma pantalla. Mismas propiedades verificadas.
SEQUENTIAL_ORANGE: Final[tuple[str, ...]] = (
    "#FEECE4",
    "#FBC8B1",
    "#F8A37B",
    "#DB8963",
    "#BB7350",
    "#9A5E41",
    "#794B35",
)

#: Rampa **ordinal** de severidad de helada (Ligera → Extrema). A diferencia de
#: la secuencial, el extremo claro sí debe superar 2:1 porque son marcas
#: discretas. Validación ordinal: monotonía PASS · ΔL adyacente PASS ·
#: contraste del extremo claro PASS (#82BAFA a 2.03:1) · tono único PASS.
ORDINAL_FROST: Final[tuple[str, ...]] = ("#82BAFA", "#5197E4", "#2E75BF", "#0C5597")

#: «Sin helada» es ausencia de fenómeno, no un grado de la escala: recibe un
#: neutro y queda fuera de la rampa ordinal.
FROST_INTENSITY_COLORS: Final[dict[str, str]] = {
    "Sin helada": NEUTRAL["200"],
    "Ligera": ORDINAL_FROST[0],
    "Moderada": ORDINAL_FROST[1],
    "Severa": ORDINAL_FROST[2],
    "Extrema": ORDINAL_FROST[3],
}

#: Rampa divergente para anomalías térmicas: naranja (cálido) ↔ azul (frío) con
#: gris neutro en el punto medio. Los dos polos son cromáticamente opuestos
#: —uno cálido, uno frío—, condición necesaria para que el signo se lea sin
#: leyenda. Claridad simétrica respecto al centro: 0.46/0.59/0.73/0.96/0.73/0.59/0.46.
DIVERGING_THERMAL: Final[tuple[str, ...]] = (
    "#8F3B05",
    "#BD6131",
    "#D99373",
    "#F1F1EE",
    "#7AAAE2",
    "#3C81CB",
    "#0658A0",
)

#: Estado del sistema de alertas. Semántica reservada: nunca se emplea como
#: color de serie. Cada uso va acompañado de icono y etiqueta textual, de modo
#: que el significado no dependa del color.
STATUS: Final[dict[str, str]] = {
    "good": "#0CA30C",
    "warning": "#FAB219",
    "serious": "#EC835A",
    "critical": "#D03B3B",
}

#: Tintes pastel de los colores de estado, para el fondo de las insignias. El
#: color saturado queda reservado al icono y al borde.
STATUS_TINT: Final[dict[str, str]] = {
    "good": "#E8F6E8",
    "warning": "#FEF4E0",
    "serious": "#FDEDE6",
    "critical": "#FBE9E9",
}

#: Colores asignados a los niveles del índice de riesgo. Es una escala
#: **ordinal**, por lo que sigue la rampa de severidad y no colores arbitrarios.
RISK_LEVEL_COLORS: Final[dict[str, str]] = {
    "Bajo": SEQUENTIAL_BLUE[1],
    "Moderado": SEQUENTIAL_BLUE[2],
    "Alto": SEQUENTIAL_BLUE[3],
    "Muy alto": SEQUENTIAL_BLUE[4],
    "Crítico": SEQUENTIAL_BLUE[6],
}

#: Temporadas del altiplano. Tres categorías nominales ⇒ tres primeras ranuras
#: de la paleta de tinta, que cumple contraste en todas ellas.
SEASON_COLORS: Final[dict[str, str]] = {
    "Lluviosa": CATEGORICAL_INK[1],
    "Transición": CATEGORICAL_INK[4],
    "Seca": CATEGORICAL_INK[0],
}

# ---------------------------------------------------------------------------
# Tokens semánticos consumidos por la interfaz y por los gráficos
# ---------------------------------------------------------------------------

PALETTE: Final[dict[str, str]] = {
    # Superficies
    "page": NEUTRAL["50"],
    "surface": NEUTRAL["0"],
    "surface_raised": NEUTRAL["25"],
    "surface_sunken": NEUTRAL["100"],
    "surface_accent": ORANGE["50"],
    "surface_accent_alt": BLUE["50"],
    # Bordes y separadores
    "border": "#E4E8EF",
    "border_strong": NEUTRAL["300"],
    "divider": NEUTRAL["200"],
    # Tinta de texto (contrastes sobre blanco: 17.7 / 7.6 / 4.8 / 2.5)
    "ink_primary": NEUTRAL["900"],
    "ink_secondary": NEUTRAL["600"],
    "ink_muted": "#6B7280",
    "ink_faint": NEUTRAL["400"],
    "ink_inverse": NEUTRAL["0"],
    # Marca
    "brand": ORANGE["200"],
    "brand_ink": ORANGE["400"],
    "brand_soft": ORANGE["50"],
    "accent": BLUE["200"],
    "accent_ink": BLUE["400"],
    "accent_soft": BLUE["50"],
    # Cromo de los gráficos
    "grid": "#EDF0F5",
    "axis": NEUTRAL["300"],
    "zeroline": NEUTRAL["400"],
    "annotation": "#6B7280",
}

#: Tipografía. Se declara una cadena de reserva completa para que el proyecto
#: mantenga su apariencia incluso sin acceso a fuentes remotas.
FONT_STACK: Final[str] = (
    'Inter, "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, '
    '"Helvetica Neue", Arial, sans-serif'
)

FONT_SIZES: Final[dict[str, int]] = {
    "chart_title": 15,
    "chart_axis_title": 12,
    "chart_tick": 11,
    "chart_legend": 12,
    "chart_annotation": 11,
    "hover": 12,
}

RADIUS: Final[dict[str, str]] = {"sm": "8px", "md": "12px", "lg": "16px", "xl": "22px", "pill": "999px"}

SHADOW: Final[dict[str, str]] = {
    "xs": "0 1px 2px rgba(17,24,39,0.04)",
    "sm": "0 1px 3px rgba(17,24,39,0.05), 0 1px 2px rgba(17,24,39,0.03)",
    "md": "0 4px 12px rgba(17,24,39,0.06), 0 1px 3px rgba(17,24,39,0.04)",
    "lg": "0 12px 28px rgba(17,24,39,0.08), 0 2px 6px rgba(17,24,39,0.04)",
    "glow_brand": "0 6px 20px rgba(219,137,99,0.20)",
}

#: Escalas continuas expuestas a Plotly, en el formato que espera la librería.
SEQUENTIAL_SCALES: Final[dict[str, list[str]]] = {
    "blue": list(SEQUENTIAL_BLUE),
    "orange": list(SEQUENTIAL_ORANGE),
    "blue_reversed": list(reversed(SEQUENTIAL_BLUE)),
    "orange_reversed": list(reversed(SEQUENTIAL_ORANGE)),
    "thermal": list(DIVERGING_THERMAL),
}

PLOTLY_TEMPLATE: Final[str] = "puno_pastel"

#: Estilo único de barra de color, reutilizado por todas las marcas continuas
#: (mapas de calor, mapas geográficos, dispersión con escala).
COLORBAR_STYLE: Final[dict[str, object]] = {
    "outlinewidth": 0,
    "thickness": 12,
    "len": 0.72,
    "y": 0.5,
    "yanchor": "middle",
    "tickfont": {"size": FONT_SIZES["chart_tick"], "color": PALETTE["ink_muted"]},
    "title": {
        "font": {"size": FONT_SIZES["chart_axis_title"], "color": PALETTE["ink_secondary"]},
        "side": "right",
    },
}

#: Configuración de la barra de herramientas de Plotly. Se retiran los botones
#: que no aportan valor analítico y se habilita la descarga en alta resolución
#: para que cualquier gráfico pueda insertarse en el informe.
PLOTLY_CONFIG: Final[dict[str, object]] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "grafico_heladas_puno"},
    "scrollZoom": False,
    "responsive": True,
}


def _axis_style(*, show_grid: bool = True, zeroline: bool = False) -> dict[str, object]:
    """Construye el estilo de un eje cartesiano.

    Args:
        show_grid: Si se dibuja la rejilla del eje.
        zeroline: Si se resalta la línea del cero.

    Returns:
        Diccionario de propiedades de eje para el layout de Plotly.
    """
    return {
        "showgrid": show_grid,
        "gridcolor": PALETTE["grid"],
        "gridwidth": 1,
        "griddash": "solid",
        "zeroline": zeroline,
        "zerolinecolor": PALETTE["zeroline"],
        "zerolinewidth": 1,
        "showline": False,
        "linecolor": PALETTE["axis"],
        "ticks": "outside",
        "ticklen": 4,
        "tickcolor": PALETTE["border"],
        "tickfont": {"size": FONT_SIZES["chart_tick"], "color": PALETTE["ink_muted"]},
        "title": {
            "font": {"size": FONT_SIZES["chart_axis_title"], "color": PALETTE["ink_secondary"]},
            "standoff": 10,
        },
        "automargin": True,
    }


def register_theme() -> None:
    """Registra la plantilla de Plotly del proyecto y la fija como predeterminada.

    La función es idempotente: puede invocarse en cada recarga de Streamlit sin
    duplicar plantillas ni provocar efectos secundarios.
    """
    template = go.layout.Template()

    template.layout = go.Layout(
        colorway=list(CATEGORICAL_INK),
        colorscale={
            "sequential": [[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
            "sequentialminus": [
                [i / (len(SEQUENTIAL_ORANGE) - 1), c] for i, c in enumerate(SEQUENTIAL_ORANGE)
            ],
            "diverging": [
                [i / (len(DIVERGING_THERMAL) - 1), c] for i, c in enumerate(DIVERGING_THERMAL)
            ],
        },
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font={"family": FONT_STACK, "size": FONT_SIZES["chart_tick"], "color": PALETTE["ink_secondary"]},
        title={
            "font": {"family": FONT_STACK, "size": FONT_SIZES["chart_title"], "color": PALETTE["ink_primary"]},
            "x": 0.0,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
            "pad": {"b": 12},
        },
        margin={"l": 8, "r": 16, "t": 56, "b": 8},
        xaxis=_axis_style(show_grid=False),
        yaxis=_axis_style(show_grid=True),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": FONT_SIZES["chart_legend"], "color": PALETTE["ink_secondary"]},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "itemsizing": "constant",
            "itemwidth": 30,
            "traceorder": "normal",
        },
        hoverlabel={
            "bgcolor": PALETTE["ink_primary"],
            "bordercolor": PALETTE["ink_primary"],
            "font": {"family": FONT_STACK, "size": FONT_SIZES["hover"], "color": PALETTE["ink_inverse"]},
            "align": "left",
        },
        hovermode="closest",
        separators=", ",
        # La rejilla y los ejes deben ceder protagonismo a las marcas de datos.
        modebar={"bgcolor": "rgba(0,0,0,0)", "color": PALETTE["ink_faint"], "activecolor": PALETTE["brand_ink"]},
        dragmode="pan",
        uniformtext={"mode": "hide", "minsize": 9},
    )

    # Especificaciones de marca: trazos finos, extremos redondeados, anillo de
    # superficie en marcas que pueden solaparse.
    template.data.bar = [
        go.Bar(
            marker={"line": {"width": 0}, "cornerradius": 4},
            textfont={"family": FONT_STACK, "size": 11, "color": PALETTE["ink_secondary"]},
            textposition="outside",
            cliponaxis=False,
        )
    ]
    template.data.scatter = [
        go.Scatter(
            marker={"size": 8, "line": {"width": 1.5, "color": PALETTE["surface"]}, "opacity": 0.85},
            line={"width": 2},
        )
    ]
    template.data.scattergl = [
        go.Scattergl(marker={"size": 7, "line": {"width": 1, "color": PALETTE["surface"]}, "opacity": 0.8})
    ]
    template.data.box = [
        go.Box(
            marker={"size": 4, "opacity": 0.55},
            line={"width": 1.5},
            fillcolor="rgba(0,0,0,0)",
            boxmean=True,
        )
    ]
    template.data.violin = [go.Violin(line={"width": 1.5}, meanline={"visible": True})]
    template.data.histogram = [go.Histogram(marker={"line": {"width": 1, "color": PALETTE["surface"]}})]
    # Separación de 2 px entre celdas: el hueco de superficie es lo que permite
    # leer la celda individual dentro de una matriz densa.
    template.data.heatmap = [
        go.Heatmap(colorscale=list(SEQUENTIAL_BLUE), xgap=2, ygap=2, colorbar=COLORBAR_STYLE)
    ]

    pio.templates[PLOTLY_TEMPLATE] = template
    pio.templates.default = PLOTLY_TEMPLATE


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convierte un color hexadecimal a notación ``rgba()`` con transparencia.

    Args:
        hex_color: Color en formato ``#RRGGBB``.
        alpha: Opacidad en el intervalo [0, 1].

    Returns:
        Cadena ``rgba(r, g, b, alpha)`` utilizable por Plotly y por CSS.

    Raises:
        ValueError: Si ``hex_color`` no tiene el formato esperado.
    """
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Se esperaba un color en formato #RRGGBB, se recibió {hex_color!r}")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def categorical_colors(n_series: int, *, all_pairs: bool = False, ink: bool = True) -> list[str]:
    """Devuelve los colores categóricos para ``n_series`` series.

    Args:
        n_series: Número de series a colorear.
        all_pairs: ``True`` si la forma gráfica permite que cualquier par de
            marcas quede adyacente (dispersión, burbujas, mapas, pequeños
            múltiplos). En ese caso rige el tope de
            :data:`MAX_ALL_PAIRS_SERIES` series.
        ink: ``True`` para la variante de tinta (contraste ≥ 3:1, obligatoria en
            trazos finos); ``False`` para el relleno pastel.

    Returns:
        Lista de colores en el orden fijo de la paleta.

    Raises:
        ValueError: Si ``n_series`` supera el tope de la paleta solicitada. El
            error es deliberado: generar un tono adicional rompería las
            garantías de accesibilidad verificadas para la paleta.
    """
    if n_series < 1:
        raise ValueError(f"n_series debe ser positivo, se recibió {n_series}")

    if all_pairs:
        if n_series > MAX_ALL_PAIRS_SERIES:
            raise ValueError(
                f"La paleta de pares completos admite como máximo {MAX_ALL_PAIRS_SERIES} "
                f"series (se solicitaron {n_series}). Agrupe las categorías menores en "
                "«Otros» o utilice pequeños múltiplos."
            )
        return list(CATEGORICAL_ALL_PAIRS[:n_series])

    source = CATEGORICAL_INK if ink else CATEGORICAL_FILL
    if n_series > len(source):
        raise ValueError(
            f"La paleta categórica admite como máximo {len(source)} series "
            f"(se solicitaron {n_series}). Agrupe las categorías menores en «Otros»."
        )
    return list(source[:n_series])
