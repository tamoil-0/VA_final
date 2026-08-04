"""Capa de configuración del proyecto.

Centraliza rutas, metadatos institucionales, parámetros de dominio y la
identidad visual del dashboard. Ningún módulo de las capas superiores
(`utils`, `components`, `pages`) debe declarar constantes propias: todas
deben importarse desde aquí para garantizar una única fuente de verdad.
"""

from config.settings import (
    AUTHORS,
    COURSE,
    DATA_SOURCE,
    FROST_INTENSITY_ORDER,
    FROST_THRESHOLDS,
    MONTH_LABELS,
    PROVINCES,
    PROJECT_PATHS,
    RESEARCH_QUESTIONS,
    SEASON_ORDER,
    STUDY_PERIOD,
    VARIABLE_DICTIONARY,
)
from config.theme import PALETTE, PLOTLY_TEMPLATE, SEQUENTIAL_SCALES, register_theme

__all__ = [
    "AUTHORS",
    "COURSE",
    "DATA_SOURCE",
    "FROST_INTENSITY_ORDER",
    "FROST_THRESHOLDS",
    "MONTH_LABELS",
    "PALETTE",
    "PLOTLY_TEMPLATE",
    "PROJECT_PATHS",
    "PROVINCES",
    "RESEARCH_QUESTIONS",
    "SEASON_ORDER",
    "SEQUENTIAL_SCALES",
    "STUDY_PERIOD",
    "VARIABLE_DICTIONARY",
    "register_theme",
]
