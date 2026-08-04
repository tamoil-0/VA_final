"""Tarjetas de sección, bloques de interpretación y hallazgos.

El componente :func:`render_interpretation` es el que materializa el apartado 9.6
de la consigna —qué representa, qué patrón se observa, qué hallazgo es relevante,
qué decisión se deriva y qué limitación presentan los datos— con una estructura
fija, de modo que ninguna sección del dashboard pueda quedar sin interpretar.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd
import streamlit as st

from utils.insights import Insight, SectionInterpretation

#: Etiquetas de los cinco puntos de interpretación exigidos por la consigna.
_INTERPRETATION_TERMS: tuple[tuple[str, str], ...] = (
    ("represents", "Qué representa"),
    ("pattern", "Patrón observado"),
    ("finding", "Hallazgo relevante"),
    ("decision", "Decisión derivada"),
    ("limitation", "Limitación del dato"),
)


def render_section_header(
    title: str, description: str, *, eyebrow: str = "", question: str = ""
) -> None:
    """Dibuja el encabezado de una sección analítica.

    Args:
        title: Título de la sección.
        description: Propósito analítico de la sección.
        eyebrow: Etiqueta superior, habitualmente el código de la pregunta.
        question: Pregunta de análisis que la sección responde. Mostrarla obliga
            a que cada visualización tenga un propósito declarado, tal como
            exige la consigna.
    """
    eyebrow_markup = f'<div class="pn-section__eyebrow">{eyebrow}</div>' if eyebrow else ""
    question_markup = f'<p class="pn-section__question">{question}</p>' if question else ""
    st.markdown(
        f"""
        <div class="pn-section">
          {eyebrow_markup}
          <h3 class="pn-section__title" style="margin-top:0">{title}</h3>
          {question_markup}
          <p class="pn-section__description">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_interpretation(interpretation: SectionInterpretation) -> None:
    """Dibuja el bloque de interpretación de cinco puntos.

    Args:
        interpretation: Interpretación redactada de la sección.
    """
    items = "".join(
        f"""
        <div class="pn-interpretation__item">
          <div class="pn-interpretation__term">{label}</div>
          <p class="pn-interpretation__text">{getattr(interpretation, attribute)}</p>
        </div>
        """
        for attribute, label in _INTERPRETATION_TERMS
    )
    st.markdown(
        f'<div class="pn-interpretation"><div class="pn-interpretation__grid">{items}</div></div>',
        unsafe_allow_html=True,
    )


def render_insight(insight: Insight) -> None:
    """Dibuja una tarjeta de hallazgo.

    Args:
        insight: Hallazgo detectado automáticamente.
    """
    recommendation = ""
    if insight.recommendation:
        recommendation = f"""
        <div class="pn-insight__recommendation">
          <span class="pn-insight__recommendation-label">Recomendación</span>
          <span>{insight.recommendation}</span>
        </div>
        """

    st.markdown(
        f"""
        <div class="pn-insight pn-insight--{insight.severity}">
          <div class="pn-insight__head">
            <span class="pn-insight__icon">{insight.icon}</span>
            <p class="pn-insight__title">{insight.title}</p>
          </div>
          <p class="pn-insight__body">{insight.body}</p>
          {recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insights(insights: list[Insight], *, columns: int = 2) -> None:
    """Dibuja una rejilla de hallazgos.

    Args:
        insights: Hallazgos a mostrar, ya ordenados por severidad.
        columns: Número de columnas de la rejilla.
    """
    if not insights:
        st.info("No se detectaron hallazgos destacables en la selección actual.")
        return

    for start in range(0, len(insights), columns):
        slots = st.columns(columns, gap="small")
        for slot, insight in zip(slots, insights[start : start + columns]):
            with slot:
                render_insight(insight)


def render_badge(label: str, severity: str = "info", *, icon: str = "") -> str:
    """Compone el marcado de una insignia de estado.

    La insignia siempre lleva icono y texto: el color por sí solo no puede portar
    significado, ni para personas con deficiencia en la visión del color ni al
    imprimir en escala de grises.

    Args:
        label: Texto de la insignia.
        severity: ``info``, ``good``, ``warning``, ``serious`` o ``critical``.
        icon: Emoji opcional que refuerza el significado.

    Returns:
        Fragmento HTML de la insignia, para incrustar en otro marcado.
    """
    icon_markup = f"{icon} " if icon else ""
    return f'<span class="pn-badge pn-badge--{severity}">{icon_markup}{label}</span>'


def render_caption(number: int, title: str, description: str = "") -> None:
    """Dibuja el pie de figura numerado de una visualización.

    Numerar y describir cada figura permite que las capturas del dashboard se
    inserten en el informe técnico conservando su referencia.

    Args:
        number: Número de la figura.
        title: Título de la figura.
        description: Aclaración metodológica o de lectura.
    """
    detail = f" {description}" if description else ""
    st.markdown(
        f'<p class="pn-caption"><span class="pn-caption__number">Figura {number}.</span> '
        f"{title}.{detail}</p>",
        unsafe_allow_html=True,
    )


def render_note(text: str) -> None:
    """Dibuja una nota metodológica discreta.

    Args:
        text: Texto de la nota.
    """
    st.markdown(f'<p class="pn-note">{text}</p>', unsafe_allow_html=True)


def render_table_view(
    frame: pd.DataFrame,
    *,
    label: str = "Ver los datos de este gráfico",
    max_rows: int = 500,
    hide_index: bool = True,
) -> None:
    """Muestra la vista de tabla asociada a una visualización.

    Este componente **no es opcional**. La paleta pastel del proyecto sitúa
    varios rellenos por debajo de la razón de contraste 3:1 frente a la
    superficie, lo que obliga a ofrecer un canal de relevo: o etiquetas visibles
    directas, o la vista de tabla. Cada gráfico que use rellenos pastel debe
    acompañarse de esta tabla.

    Args:
        frame: Datos que sostienen la visualización.
        label: Texto del desplegable.
        max_rows: Número máximo de filas a mostrar.
        hide_index: Si se oculta el índice del dataframe.
    """
    if frame is None or frame.empty:
        return

    with st.expander(label, expanded=False):
        truncated = len(frame) > max_rows
        st.dataframe(
            frame.head(max_rows),
            width="stretch",
            hide_index=hide_index,
        )
        if truncated:
            st.caption(
                f"Se muestran las primeras {max_rows:,} de {len(frame):,} filas. "
                "Use el panel de descargas para obtener el conjunto completo."
            )


@contextmanager
def card() -> Iterator[None]:
    """Envuelve un bloque de contenido en una tarjeta con borde y sombra.

    Yields:
        Control al bloque interno, que se dibuja dentro del contenedor.
    """
    container = st.container(border=True)
    with container:
        yield


def render_metric_grid(metrics: dict[str, str], *, columns: int = 4) -> None:
    """Muestra una rejilla compacta de pares etiqueta-valor.

    Es el formato adecuado para resultados de una prueba estadística, donde hacen
    falta varias cifras pequeñas y no tarjetas grandes de indicador.

    Args:
        metrics: Diccionario ``{etiqueta: valor_formateado}``.
        columns: Número de columnas.
    """
    if not metrics:
        return

    items = list(metrics.items())
    for start in range(0, len(items), columns):
        slots = st.columns(columns, gap="small")
        for slot, (label, value) in zip(slots, items[start : start + columns]):
            with slot:
                st.markdown(
                    f"""
                    <div class="pn-kpi" style="padding:12px 14px">
                      <span class="pn-kpi__label">{label}</span>
                      <div class="pn-kpi__value" style="font-size:1.25rem">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
