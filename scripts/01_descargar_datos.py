"""Paso 1 del flujo de datos: descarga de los conjuntos originales.

Uso::

    python scripts/01_descargar_datos.py            # ambas fuentes
    python scripts/01_descargar_datos.py --solo-primaria

Produce:
    * ``data/raw/era5_NN_<capital>.json``           — respuesta íntegra por punto.
    * ``data/raw/unidades_nativas_era5.json``       — unidades declaradas por la fuente.
    * ``data/datos_originales.csv``                 — dataset primario sin procesar.
    * ``data/datos_originales_validacion_merra2.csv`` — fuente secundaria de control.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (  # noqa: E402
    DATA_SOURCE,
    DATA_SOURCE_SECONDARY,
    ERA5_DAILY_VARIABLES,
    PROVINCES,
    STUDY_PERIOD,
)
from utils.data_acquisition import download_primary_dataset, download_validation_dataset  # noqa: E402


def _print_header() -> None:
    """Imprime la ficha de procedencia de las fuentes consultadas."""
    print("=" * 78)
    print("DESCARGA DE LOS CONJUNTOS DE DATOS ORIGINALES")
    print("=" * 78)
    print("FUENTE PRIMARIA")
    print(f"  Nombre     : {DATA_SOURCE.name}")
    print(f"  Institución: {DATA_SOURCE.organization}")
    print(f"  Modelo     : {DATA_SOURCE.model}")
    print(f"  Resolución : {DATA_SOURCE.spatial_resolution}")
    print(f"  Variables  : {len(ERA5_DAILY_VARIABLES)} variables diarias")
    print("FUENTE SECUNDARIA (validación cruzada)")
    print(f"  Nombre     : {DATA_SOURCE_SECONDARY.name}")
    print(f"  Modelo     : {DATA_SOURCE_SECONDARY.model}")
    print(f"  Resolución : {DATA_SOURCE_SECONDARY.spatial_resolution}")
    print("COBERTURA")
    print(f"  Periodo    : {STUDY_PERIOD.start} a {STUDY_PERIOD.end} ({STUDY_PERIOD.n_years} años)")
    print(f"  Puntos     : {len(PROVINCES)} capitales provinciales de la región Puno")
    print("-" * 78)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta la descarga de una o ambas fuentes.

    Args:
        argv: Argumentos de línea de comandos. ``None`` usa ``sys.argv``.

    Returns:
        ``0`` si la descarga concluyó correctamente.
    """
    parser = argparse.ArgumentParser(description="Descarga los datasets originales del proyecto.")
    parser.add_argument(
        "--solo-primaria",
        action="store_true",
        help="Descarga únicamente la fuente primaria y omite la de validación.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    _print_header()

    primary = download_primary_dataset()
    print("-" * 78)
    print("FUENTE PRIMARIA")
    print(f"  Filas      : {len(primary):,}")
    print(f"  Columnas   : {len(primary.columns)}")
    print(f"  Provincias : {primary['provincia'].nunique()}")
    print(f"  Fechas     : {primary['fecha_iso'].min()} – {primary['fecha_iso'].max()}")

    elevations = (
        primary.groupby("provincia", observed=True)["elevacion_malla_m"].first().sort_values()
    )
    print(f"  Elevaciones de malla distintas: {elevations.nunique()} de {len(elevations)} provincias")
    print(f"    mínima  → {elevations.index[0]}: {elevations.iloc[0]:.0f} msnm")
    print(f"    máxima  → {elevations.index[-1]}: {elevations.iloc[-1]:.0f} msnm")

    if not args.solo_primaria:
        validation = download_validation_dataset()
        print("-" * 78)
        print("FUENTE SECUNDARIA")
        print(f"  Filas      : {len(validation):,}")
        print(f"  Parámetros : {[c for c in validation.columns if c not in ('provincia', 'YYYYMMDD')]}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
