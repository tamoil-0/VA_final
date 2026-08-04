"""Estructura de página: configuración, estilos, encabezado institucional y pie.

Toda página del dashboard comienza invocando :func:`setup_page`, que fija la
configuración de Streamlit, registra la plantilla de Plotly e inyecta la hoja de
estilo. Concentrar esa secuencia en una sola función evita que una página quede
con el tema a medio aplicar.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from config.settings import (
    APP_CONFIG,
    AUTHORS,
    COURSE,
    DATA_SOURCE,
    PROBLEM_STATEMENT,
    PROJECT_PATHS,
    RESEARCH_QUESTIONS,
    STUDY_PERIOD,
)
from config.theme import register_theme
from utils.io import load_metadata


@lru_cache(maxsize=8)
def _encode_image(path_text: str) -> str | None:
    """Codifica una imagen en base64 para incrustarla en el marcado.

    Se incrusta en lugar de servirse como archivo estático para que el
    encabezado se pinte en el primer fotograma, sin una petición adicional que
    provoque un salto visible del diseño.

    Args:
        path_text: Ruta del archivo de imagen, en texto.

    Returns:
        La cadena base64, o ``None`` si el archivo no existe.
    """
    path = Path(path_text)
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


@lru_cache(maxsize=1)
def _load_stylesheet() -> str:
    """Lee la hoja de estilo principal del proyecto.

    Returns:
        Contenido CSS, o cadena vacía si el archivo no está disponible.
    """
    path = PROJECT_PATHS.styles / "main.css"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def setup_page(page_title: str, *, page_icon: str | None = None) -> None:
    """Configura la página y aplica la identidad visual del proyecto.

    Args:
        page_title: Título de la pestaña del navegador.
        page_icon: Icono de la pestaña. Si es ``None`` se usa el del proyecto.
    """
    st.set_page_config(
        page_title=f"{page_title} · {COURSE['university_short']}",
        page_icon=page_icon or APP_CONFIG["page_icon"],
        layout=APP_CONFIG["layout"],
        initial_sidebar_state=APP_CONFIG["initial_sidebar_state"],
        menu_items=APP_CONFIG["menu_items"],
    )

    register_theme()

    stylesheet = _load_stylesheet()
    if stylesheet:
        st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)

    # La familia tipográfica se solicita a un servicio externo, pero la cadena de
    # reserva declarada en el CSS garantiza el mismo diseño sin conexión.
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" '
        'rel="stylesheet">',
        unsafe_allow_html=True,
    )


def render_header(*, compact: bool = False) -> None:
    """Dibuja el encabezado institucional con la identificación del proyecto.

    Cumple el apartado 9.1 de la consigna: título, descripción del problema,
    integrantes, fuente y periodo de los datos, y fecha de actualización.

    Args:
        compact: Si se omite la descripción extensa del problema, para las
            páginas interiores donde el contexto ya se ha establecido.
    """
    unap_logo = _encode_image(str(PROJECT_PATHS.logos / "logo_unap.png"))
    episi_logo = _encode_image(str(PROJECT_PATHS.logos / "logo_episi.png"))

    logos = "".join(
        f'<img src="data:image/png;base64,{logo}" alt="{alt}">'
        for logo, alt in ((unap_logo, "Escudo de la Universidad Nacional del Altiplano"),
                          (episi_logo, "Logotipo de la Escuela Profesional de Ingeniería de Sistemas"))
        if logo
    )

    metadata = load_metadata()
    updated_at = str(metadata.get("generado_en", ""))[:10] or "—"
    member_names = ", ".join(author.display_name for author in AUTHORS)

    description = (
        ""
        if compact
        else f'<p class="pn-header__subtitle">{PROBLEM_STATEMENT}</p>'
    )

    st.markdown(
        f"""
        <div class="pn-header">
          <div class="pn-header__top">
            <div class="pn-header__logos">{logos}</div>
            <div class="pn-header__institution">
              <p class="pn-header__university">{COURSE['university']}</p>
              <p class="pn-header__unit">{COURSE['faculty']}<br>{COURSE['school']}</p>
            </div>
          </div>
          <h1 class="pn-header__title">{COURSE['project_title']}</h1>
          {description}
          <div class="pn-header__meta">
            <span class="pn-chip pn-chip--brand">
              <span class="pn-chip__label">Curso</span>
              <span class="pn-chip__value">{COURSE['course']} · {COURSE['course_code']}</span>
            </span>
            <span class="pn-chip">
              <span class="pn-chip__label">Docente</span>
              <span class="pn-chip__value">{COURSE['professor']}</span>
            </span>
            <span class="pn-chip">
              <span class="pn-chip__label">Semestre</span>
              <span class="pn-chip__value">{COURSE['semester']} · Ciclo {COURSE['cycle']}</span>
            </span>
            <span class="pn-chip pn-chip--accent">
              <span class="pn-chip__label">Fuente</span>
              <span class="pn-chip__value">ERA5-Land · Copernicus (C3S)</span>
            </span>
            <span class="pn-chip pn-chip--accent">
              <span class="pn-chip__label">Periodo</span>
              <span class="pn-chip__value">{STUDY_PERIOD.label}</span>
            </span>
            <span class="pn-chip">
              <span class="pn-chip__label">Ámbito</span>
              <span class="pn-chip__value">13 provincias de Puno</span>
            </span>
            <span class="pn-chip">
              <span class="pn-chip__label">Actualizado</span>
              <span class="pn-chip__value">{updated_at}</span>
            </span>
          </div>
          <div class="pn-header__meta" style="margin-top:8px">
            <span class="pn-chip">
              <span class="pn-chip__label">Integrantes</span>
              <span class="pn-chip__value">{member_names}</span>
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_title(title: str, subtitle: str, *, eyebrow: str = "") -> None:
    """Dibuja el título de una página interior.

    Args:
        title: Título de la página.
        subtitle: Descripción de su propósito analítico.
        eyebrow: Etiqueta superior opcional, para situar la página en el flujo.
    """
    eyebrow_markup = f'<div class="pn-section__eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="pn-section">
          {eyebrow_markup}
          <h2 class="pn-section__title" style="font-size:1.42rem;margin-top:0">{title}</h2>
          <p class="pn-section__description">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_questions(*, page_filter: str | None = None) -> None:
    """Muestra las preguntas de análisis y el método con que se responden.

    Args:
        page_filter: Si se indica, sólo se listan las preguntas cuya página
            contiene ese texto.
    """
    questions = [
        question
        for question in RESEARCH_QUESTIONS
        if page_filter is None or page_filter.lower() in question.page.lower()
    ]
    if not questions:
        return

    rows = "".join(
        f"""
        <div class="pn-question">
          <div class="pn-question__code">{question.code}</div>
          <div>
            <p class="pn-question__text">{question.question}</p>
            <p class="pn-question__method">{question.method}</p>
          </div>
        </div>
        """
        for question in questions
    )
    st.markdown(
        f"""
        <div class="pn-section">
          <div class="pn-section__eyebrow">Preguntas de análisis</div>
          {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_team() -> None:
    """Muestra las tarjetas del equipo con el aporte de cada integrante.

    La consigna exige que cada integrante participe en el procesamiento, la
    programación, el análisis y la sustentación; documentar aquí los aportes
    concretos hace verificable ese requisito.
    """
    cards = "".join(
        f"""
        <div class="pn-team__card">
          <div class="pn-team__avatar">{_initials(author.display_name)}</div>
          <p class="pn-team__name">{author.display_name}</p>
          <p class="pn-team__role">{author.role}</p>
          <ul class="pn-team__contributions">
            {''.join(f'<li>{contribution}</li>' for contribution in author.contributions)}
          </ul>
        </div>
        """
        for author in AUTHORS
    )
    st.markdown(f'<div class="pn-team">{cards}</div>', unsafe_allow_html=True)


def _initials(full_name: str) -> str:
    """Extrae las iniciales de un nombre completo.

    Args:
        full_name: Nombre en orden natural.

    Returns:
        Las dos primeras iniciales en mayúscula.
    """
    parts = [part for part in full_name.split() if part]
    return "".join(part[0].upper() for part in parts[:2])


def render_footer() -> None:
    """Dibuja el pie institucional con la atribución de la fuente de datos."""
    st.markdown(
        f"""
        <div class="pn-footer">
          <div>
            <strong>{COURSE['project_title']}</strong><br>
            {COURSE['course']} ({COURSE['course_code']}) · {COURSE['school']}<br>
            {COURSE['university']} · {COURSE['city']}, {COURSE['year']}
          </div>
          <div>
            <strong>Fuente de datos</strong><br>
            {DATA_SOURCE.organization}<br>
            {DATA_SOURCE.model} · {DATA_SOURCE.spatial_resolution}
          </div>
          <div>
            <strong>Equipo</strong><br>
            {'<br>'.join(author.display_name for author in AUTHORS)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    """Inserta un separador horizontal discreto."""
    st.markdown('<hr class="pn-divider">', unsafe_allow_html=True)


def spacer(size: str = "md") -> None:
    """Inserta espacio vertical.

    Args:
        size: ``"sm"``, ``"md"`` o ``"lg"``.
    """
    st.markdown(f'<div class="pn-spacer-{size}"></div>', unsafe_allow_html=True)


def render_empty_state(title: str, message: str, *, icon: str = "🔍") -> None:
    """Muestra un estado vacío cuando la selección no devuelve datos.

    Args:
        title: Título del mensaje.
        message: Explicación e indicación de qué hacer a continuación.
        icon: Emoji ilustrativo.
    """
    st.markdown(
        f"""
        <div class="pn-empty">
          <div class="pn-empty__icon">{icon}</div>
          <p class="pn-empty__title">{title}</p>
          <p class="pn-empty__text">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
