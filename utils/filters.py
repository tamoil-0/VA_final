"""Estado de filtros global y su aplicación sobre el dataframe.

El estado de filtros se modela como un objeto inmutable
(:class:`FilterState`) que las páginas reciben ya construido. Esa decisión tiene
dos consecuencias deseables:

* **Filtrado cruzado real.** Un único estado se comparte entre las cinco páginas
  a través de ``st.session_state``, de modo que la selección hecha en la vista
  general sigue vigente al navegar al análisis multidimensional.
* **Trazabilidad.** El objeto sabe describirse a sí mismo
  (:meth:`FilterState.describe`), lo que permite anotar cada exportación y cada
  captura con el subconjunto exacto que la generó.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import pandas as pd

from config.settings import FROST_THRESHOLDS


@dataclass(frozen=True)
class FilterState:
    """Selección activa del usuario en el panel de filtros.

    Attributes:
        date_range: Intervalo de fechas ``(inicio, fin)`` inclusivo.
        provinces: Provincias seleccionadas. Vacío significa «todas».
        ecological_tiers: Pisos ecológicos seleccionados.
        seasons: Temporadas seleccionadas.
        min_temperature_range: Intervalo admitido de temperatura mínima, en °C.
        altitude_range: Intervalo admitido de altitud, en msnm.
        frost_threshold: Umbral activo, ``"meteorologica"`` o ``"agronomica"``.
        only_frost_days: Si se restringe el análisis a los días con helada.
        exclude_suspicious: Si se excluyen los registros marcados por el control
            de calidad de la precipitación.
        analysis_variable: Variable numérica que dirige los gráficos configurables.
    """

    date_range: tuple[date, date]
    provinces: tuple[str, ...] = ()
    ecological_tiers: tuple[str, ...] = ()
    seasons: tuple[str, ...] = ()
    min_temperature_range: tuple[float, float] | None = None
    altitude_range: tuple[float, float] | None = None
    frost_threshold: str = "meteorologica"
    only_frost_days: bool = False
    exclude_suspicious: bool = False
    analysis_variable: str = "temperatura_minima"
    _defaults: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def frost_column(self) -> str:
        """Nombre de la columna booleana que corresponde al umbral activo."""
        return "helada" if self.frost_threshold == "meteorologica" else "helada_agronomica"

    @property
    def frost_threshold_value(self) -> float:
        """Valor en °C del umbral de helada activo."""
        return FROST_THRESHOLDS[self.frost_threshold]

    def describe(self) -> str:
        """Resume la selección activa en una línea legible.

        Returns:
            Descripción compacta del subconjunto, apta para anotar exportaciones
            y pies de figura.
        """
        parts = [f"{self.date_range[0]:%d/%m/%Y}–{self.date_range[1]:%d/%m/%Y}"]

        if self.provinces:
            parts.append(
                f"{len(self.provinces)} provincia(s)"
                if len(self.provinces) > 3
                else ", ".join(self.provinces)
            )
        else:
            parts.append("todas las provincias")

        if self.seasons:
            parts.append("temporada " + "/".join(self.seasons))
        if self.ecological_tiers:
            parts.append("piso " + "/".join(self.ecological_tiers))
        if self.only_frost_days:
            parts.append(f"sólo días con helada (Tmín ≤ {self.frost_threshold_value:.0f} °C)")
        if self.exclude_suspicious:
            parts.append("sin registros de precipitación sospechosa")
        return " · ".join(parts)

    def is_pristine(self) -> bool:
        """Indica si la selección coincide con el estado inicial.

        Returns:
            ``True`` si el usuario no ha modificado ningún filtro.
        """
        if not self._defaults:
            return False
        return all(
            getattr(self, name) == value
            for name, value in self._defaults.items()
            if name != "_defaults"
        )

    def with_defaults(self, defaults: dict[str, Any]) -> FilterState:
        """Devuelve una copia que recuerda cuál era el estado inicial.

        Args:
            defaults: Valores por defecto de cada filtro.

        Returns:
            Nueva instancia con los valores de referencia incorporados.
        """
        return replace(self, _defaults=defaults)


def apply_filters(frame: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    """Aplica el estado de filtros al dataframe.

    El orden de aplicación es deliberado: primero los filtros de alta
    selectividad (fecha y provincia), que descartan la mayor parte de las filas
    con una sola comparación, y después los de rango continuo. Así se reduce el
    número de filas sobre las que operan las comparaciones más costosas.

    Args:
        frame: Dataset analítico completo.
        state: Selección activa del usuario.

    Returns:
        Copia filtrada del dataframe. Nunca se modifica el original, porque está
        en la caché de Streamlit y es compartido por todas las páginas.
    """
    start, end = state.date_range
    mask = frame["fecha"].between(pd.Timestamp(start), pd.Timestamp(end))

    if state.provinces:
        mask &= frame["provincia"].isin(state.provinces)
    if state.ecological_tiers:
        mask &= frame["piso_ecologico"].isin(state.ecological_tiers)
    if state.seasons:
        mask &= frame["temporada"].isin(state.seasons)

    if state.min_temperature_range is not None:
        low, high = state.min_temperature_range
        mask &= frame["temperatura_minima"].between(low, high)

    if state.altitude_range is not None:
        low, high = state.altitude_range
        mask &= frame["altitud"].between(low, high)

    if state.only_frost_days:
        mask &= frame[state.frost_column]

    if state.exclude_suspicious and "precipitacion_sospechosa" in frame.columns:
        mask &= ~frame["precipitacion_sospechosa"]

    return frame.loc[mask].copy()


def filter_impact(total_rows: int, filtered_rows: int) -> dict[str, float | int]:
    """Cuantifica el efecto de la selección sobre el volumen de datos.

    Args:
        total_rows: Filas del dataset completo.
        filtered_rows: Filas que sobreviven al filtrado.

    Returns:
        Diccionario con las filas retenidas, descartadas y el porcentaje
        retenido, para informar al usuario del alcance de su selección.
    """
    retained_share = 100.0 * filtered_rows / total_rows if total_rows else 0.0
    return {
        "retenidas": filtered_rows,
        "descartadas": total_rows - filtered_rows,
        "porcentaje_retenido": round(retained_share, 2),
    }


def build_default_state(frame: pd.DataFrame) -> FilterState:
    """Construye el estado inicial de filtros a partir del dataset.

    Los valores por defecto se derivan de los datos y no se codifican a mano:
    así el dashboard sigue siendo correcto si el periodo de estudio se amplía.

    Args:
        frame: Dataset analítico completo.

    Returns:
        Estado de filtros inicial, sin ninguna restricción activa.
    """
    minimum_date = frame["fecha"].min().date()
    maximum_date = frame["fecha"].max().date()

    defaults: dict[str, Any] = {
        "date_range": (minimum_date, maximum_date),
        "provinces": (),
        "ecological_tiers": (),
        "seasons": (),
        "min_temperature_range": (
            float(frame["temperatura_minima"].min()),
            float(frame["temperatura_minima"].max()),
        ),
        "altitude_range": (float(frame["altitud"].min()), float(frame["altitud"].max())),
        "frost_threshold": "meteorologica",
        "only_frost_days": False,
        "exclude_suspicious": False,
        "analysis_variable": "temperatura_minima",
    }
    return FilterState(**defaults).with_defaults(defaults)
