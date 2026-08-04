"""Formato de números, fechas y unidades para la interfaz.

Centralizar el formato evita la inconsistencia más visible de un dashboard: que
la misma magnitud aparezca como ``2.5``, ``2,50`` y ``2.50 °C`` en tres tarjetas
distintas. Todas las funciones emplean la convención española —coma decimal y
punto de millar— y el espacio fino irrompible antes de la unidad, como exige la
norma tipográfica del Sistema Internacional.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from config.settings import MONTH_LABELS, VARIABLE_DICTIONARY

#: Espacio fino irrompible: separa la cifra de su unidad sin permitir que el
#: salto de línea las divida.
THIN_NBSP: str = " "

#: Índice de especificaciones de variable, para resolver etiquetas y unidades.
_SPECS = {spec.name: spec for spec in VARIABLE_DICTIONARY}


def format_number(
    value: float | int | None,
    *,
    decimals: int = 1,
    unit: str = "",
    signed: bool = False,
    thousands: bool = True,
) -> str:
    """Formatea un número con la convención española.

    Args:
        value: Valor a formatear. ``None`` y ``NaN`` devuelven un guion largo.
        decimals: Número de cifras decimales.
        unit: Unidad que se añade tras un espacio fino.
        signed: Si se fuerza el signo explícito, útil en variaciones.
        thousands: Si se aplica el separador de millares.

    Returns:
        Cadena formateada, o ``«—»`` si el valor no está disponible.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"

    specification = f"{'+' if signed else ''}{',' if thousands else ''}.{decimals}f"
    text = format(float(value), specification)
    # Se intercambian los separadores anglosajones por los españoles en un solo
    # paso, usando un marcador temporal para no sobrescribir el resultado.
    text = text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"{text}{THIN_NBSP}{unit}" if unit else text


def format_integer(value: float | int | None, *, unit: str = "") -> str:
    """Formatea un entero con separador de millares.

    Args:
        value: Valor a formatear.
        unit: Unidad que se añade tras un espacio fino.

    Returns:
        Cadena formateada sin decimales.
    """
    return format_number(value, decimals=0, unit=unit)


def format_percent(value: float | None, *, decimals: int = 1, signed: bool = False) -> str:
    """Formatea una proporción ya expresada en porcentaje.

    Args:
        value: Valor en la escala 0–100.
        decimals: Número de cifras decimales.
        signed: Si se fuerza el signo explícito.

    Returns:
        Cadena con el símbolo de porcentaje.
    """
    return format_number(value, decimals=decimals, unit="%", signed=signed)


def format_temperature(value: float | None, *, decimals: int = 1, signed: bool = False) -> str:
    """Formatea una temperatura en grados Celsius.

    Args:
        value: Temperatura en °C.
        decimals: Número de cifras decimales.
        signed: Si se fuerza el signo explícito.

    Returns:
        Cadena con la unidad de temperatura.
    """
    return format_number(value, decimals=decimals, unit="°C", signed=signed)


def format_date(value: date | datetime | pd.Timestamp | None, *, long: bool = False) -> str:
    """Formatea una fecha en español.

    Args:
        value: Fecha a formatear.
        long: Si se usa el formato extenso ``«12 de julio de 2023»`` en lugar de
            ``«12/07/2023»``.

    Returns:
        Cadena con la fecha formateada.
    """
    if value is None or pd.isna(value):
        return "—"
    stamp = pd.Timestamp(value)
    if long:
        return f"{stamp.day} de {MONTH_LABELS[stamp.month].lower()} de {stamp.year}"
    return stamp.strftime("%d/%m/%Y")


def format_date_range(start: date | datetime | pd.Timestamp, end: date | datetime | pd.Timestamp) -> str:
    """Formatea un intervalo de fechas de forma compacta.

    Args:
        start: Fecha inicial.
        end: Fecha final.

    Returns:
        Cadena con el intervalo, empleando la raya como separador.
    """
    return f"{format_date(start)} – {format_date(end)}"


def variable_label(name: str, *, with_unit: bool = True) -> str:
    """Devuelve la etiqueta legible de una variable.

    Args:
        name: Nombre técnico de la columna.
        with_unit: Si se añade la unidad entre paréntesis.

    Returns:
        Etiqueta lista para mostrarse en un eje, un selector o un título.
    """
    specification = _SPECS.get(name)
    if specification is None:
        return name.replace("_", " ").capitalize()
    if with_unit and specification.unit:
        return f"{specification.label} ({specification.unit})"
    return specification.label


def variable_unit(name: str) -> str:
    """Devuelve la unidad de una variable, o cadena vacía si es adimensional.

    Args:
        name: Nombre técnico de la columna.

    Returns:
        Unidad de medida de la variable.
    """
    specification = _SPECS.get(name)
    return specification.unit if specification else ""


def variable_description(name: str) -> str:
    """Devuelve la definición operativa de una variable.

    Args:
        name: Nombre técnico de la columna.

    Returns:
        Descripción documentada de la variable.
    """
    specification = _SPECS.get(name)
    return specification.description if specification else "Variable auxiliar del modelo de datos."


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Concuerda un sustantivo con su cantidad.

    Args:
        count: Cantidad que determina el número gramatical.
        singular: Forma singular del sustantivo.
        plural: Forma plural. Si se omite se añade una ``s``.

    Returns:
        Cadena con la cantidad y el sustantivo concordado.
    """
    word = singular if abs(count) == 1 else (plural or f"{singular}s")
    return f"{format_integer(count)} {word}"


def format_pvalue(p_value: float | None) -> str:
    """Formatea un valor p con la convención habitual en publicación científica.

    Args:
        p_value: Valor p a formatear.

    Returns:
        ``«p < 0,001»`` cuando el valor es muy pequeño; en otro caso, el valor
        con tres decimales.
    """
    if p_value is None or pd.isna(p_value):
        return "p = —"
    if p_value < 0.001:
        return "p < 0,001"
    return f"p = {format_number(p_value, decimals=3, thousands=False)}"


def significance_marker(p_value: float | None, *, alpha: float = 0.05) -> str:
    """Devuelve el marcador textual de significancia estadística.

    Args:
        p_value: Valor p de la prueba.
        alpha: Nivel de significancia adoptado.

    Returns:
        Texto que indica si el resultado es significativo al nivel dado.
    """
    if p_value is None or pd.isna(p_value):
        return "no evaluable"
    return "significativo" if p_value < alpha else "no significativo"
