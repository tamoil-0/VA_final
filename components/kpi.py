"""Tarjetas de indicador clave.

Se implementan con marcado propio en lugar de ``st.metric`` por dos razones que
el componente nativo no cubre:

1. **La variación adversa no coincide con el signo.** En este dominio, un
   aumento de los días de helada es desfavorable y una caída de la temperatura
   mínima también lo es. ``st.metric`` colorea según el signo, lo que invertiría
   la lectura en la mitad de los indicadores.
2. **Ayuda contextual y leyenda secundaria.** Cada indicador necesita su
   definición operativa y un dato de contexto, para que la cifra no se
   interprete fuera de escala.
"""

from __future__ import annotations

import streamlit as st

from utils.metrics import Kpi

#: Glifos direccionales de la variación. Se acompañan siempre de la cifra, de
#: modo que la dirección nunca dependa sólo de la forma del icono.
_ARROWS: dict[str, str] = {"up": "▲", "down": "▼", "flat": "—"}


def render_kpi_card(kpi: Kpi) -> None:
    """Dibuja una tarjeta de indicador.

    Args:
        kpi: Indicador ya calculado y formateado.
    """
    delta_markup = ""
    if kpi.delta:
        variant = (
            "neutral"
            if kpi.delta_direction == "flat"
            else ("adverse" if kpi.delta_is_adverse else "favourable")
        )
        arrow = _ARROWS.get(kpi.delta_direction, "—")
        delta_markup = (
            f'<span class="pn-kpi__delta pn-kpi__delta--{variant}">{arrow} {kpi.delta}</span>'
        )

    # Los valores no numéricos (un nombre de provincia) necesitan un cuerpo
    # tipográfico menor para no desbordar la tarjeta.
    value_class = "pn-kpi__value"
    if not any(character.isdigit() for character in kpi.value):
        value_class += " pn-kpi__value--text"

    st.markdown(
        f"""
        <div class="pn-kpi" title="{kpi.help_text}">
          <div class="pn-kpi__head">
            <span class="pn-kpi__label">{kpi.label}</span>
            <span class="pn-kpi__icon">{kpi.icon}</span>
          </div>
          <div class="{value_class}">{kpi.value}</div>
          {delta_markup}
          <div class="pn-kpi__caption">{kpi.caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(kpis: list[Kpi], *, columns: int = 4) -> None:
    """Dibuja una fila de tarjetas de indicador.

    Args:
        kpis: Indicadores a mostrar.
        columns: Número de tarjetas por fila.
    """
    if not kpis:
        return

    for start in range(0, len(kpis), columns):
        block = kpis[start : start + columns]
        # Se crean siempre `columns` huecos para que la última fila incompleta no
        # estire sus tarjetas y rompa la rejilla.
        slots = st.columns(columns, gap="small")
        for slot, kpi in zip(slots, block):
            with slot:
                render_kpi_card(kpi)
        if start + columns < len(kpis):
            st.markdown('<div class="pn-spacer-sm"></div>', unsafe_allow_html=True)


def render_kpi_help(kpis: list[Kpi]) -> None:
    """Muestra la definición operativa de cada indicador en un desplegable.

    La ayuda por superposición del cursor no es accesible por teclado ni visible
    al imprimir; este bloque garantiza que la definición esté siempre disponible.

    Args:
        kpis: Indicadores cuya definición se documenta.
    """
    if not kpis:
        return

    with st.expander("Definición de cada indicador", expanded=False):
        for kpi in kpis:
            st.markdown(f"**{kpi.icon} {kpi.label}** — {kpi.help_text}")
