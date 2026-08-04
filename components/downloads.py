"""Exportaciones trazables del dashboard en CSV, Excel, Markdown y PNG."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import DATA_SOURCE
from utils.filters import FilterState
from utils.insights import build_executive_summary
from utils.io import (
    dataframe_to_csv_bytes,
    dataframes_to_excel_bytes,
    figure_to_png_bytes,
    load_data_dictionary,
    timestamped_filename,
)
from utils.metrics import (
    aggregate_annual,
    aggregate_by_province,
    aggregate_monthly,
    compute_kpis,
)


def render_csv_button(
    frame: pd.DataFrame,
    *,
    filename_stem: str,
    label: str = "Descargar CSV (datos filtrados)",
    key: str | None = None,
) -> None:
    """Muestra un botón de descarga CSV compatible con Excel."""
    st.download_button(
        label,
        data=dataframe_to_csv_bytes(frame),
        file_name=timestamped_filename(filename_stem, "csv"),
        mime="text/csv",
        key=key,
        width="stretch",
    )


def render_excel_button(
    sheets: dict[str, pd.DataFrame],
    *,
    filename_stem: str,
    label: str = "Descargar Excel (multi-hoja)",
    key: str | None = None,
) -> None:
    """Muestra un botón de descarga para un libro Excel multihoja."""
    st.download_button(
        label,
        data=dataframes_to_excel_bytes(sheets),
        file_name=timestamped_filename(filename_stem, "xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
        width="stretch",
    )


def render_figure_download(
    figure: Any,
    *,
    filename_stem: str,
    label: str = "Descargar figura PNG",
    key: str | None = None,
) -> None:
    """Permite descargar una figura a alta resolución cuando Kaleido está disponible."""
    image = figure_to_png_bytes(figure, scale=3)
    if image is None:
        st.caption("La exportación PNG no está disponible en este entorno.")
        return
    st.download_button(
        label,
        data=image,
        file_name=timestamped_filename(filename_stem, "png"),
        mime="image/png",
        key=key,
        width="stretch",
    )


def build_report_summary(filtered: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    """Construye una ficha ejecutiva de la selección activa."""
    province = aggregate_by_province(filtered, frost_column=state.frost_column)
    rows: list[dict[str, object]] = [
        {"Indicador": "Selección", "Valor": state.describe()},
        {"Indicador": "Registros", "Valor": len(filtered)},
        {"Indicador": "Provincias", "Valor": filtered["provincia"].nunique()},
        {"Indicador": "Días", "Valor": filtered["fecha"].nunique()},
        {
            "Indicador": "Tasa de helada",
            "Valor": f"{filtered[state.frost_column].mean() * 100:.1f} %",
        },
        {
            "Indicador": "Temperatura mínima media",
            "Valor": f"{filtered['temperatura_minima'].mean():.2f} °C",
        },
        {
            "Indicador": "Provincia con mayor incidencia",
            "Valor": province.iloc[0]["provincia"] if not province.empty else "—",
        },
        {"Indicador": "Fuente", "Valor": DATA_SOURCE.name},
    ]
    return pd.DataFrame(rows)


def _metadata_sheet(filtered: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "campo": [
                "seleccion",
                "generado_en",
                "registros",
                "umbral_helada_c",
                "fuente",
                "cita_apa",
            ],
            "valor": [
                state.describe(),
                datetime.now().astimezone().isoformat(timespec="seconds"),
                str(len(filtered)),
                str(state.frost_threshold_value),
                DATA_SOURCE.name,
                DATA_SOURCE.citation_apa,
            ],
        }
    )


def render_download_panel(
    filtered: pd.DataFrame,
    state: FilterState,
    *,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
    key_prefix: str = "",
    include_dictionary: bool = True,
) -> None:
    """Dibuja el panel unificado de exportación con metadatos de trazabilidad."""
    if filtered.empty:
        return

    st.markdown("#### Exportar la selección")
    st.caption(
        "Los archivos conservan los filtros activos, la fecha de generación y la cita de la fuente."
    )
    sheets: dict[str, pd.DataFrame] = {
        "datos_filtrados": filtered,
        "resumen_provincial": aggregate_by_province(
            filtered, frost_column=state.frost_column
        ),
        "resumen_mensual": aggregate_monthly(filtered, frost_column=state.frost_column),
        "resumen_anual": aggregate_annual(filtered, frost_column=state.frost_column),
        "kpis": pd.DataFrame(asdict(item) for item in compute_kpis(
            filtered, frost_column=state.frost_column
        )),
        "metadatos_seleccion": _metadata_sheet(filtered, state),
    }
    if include_dictionary:
        sheets["diccionario_datos"] = load_data_dictionary()
    if extra_sheets:
        sheets.update({name[:31]: value for name, value in extra_sheets.items() if value is not None})

    prefix = f"{key_prefix}_" if key_prefix else ""
    first, second, third = st.columns(3, gap="small")
    with first:
        render_csv_button(
            filtered,
            filename_stem="heladas_puno_filtrado",
            key=f"{prefix}download_csv",
        )
    with second:
        # Un libro con decenas de miles de filas tarda varios segundos en
        # construirse. Se prepara bajo demanda para que mover un filtro no
        # bloquee innecesariamente toda la interfaz.
        excel_state_key = f"pn_excel_bytes_{prefix or 'default'}_{abs(hash(state))}"
        if excel_state_key not in st.session_state:
            if st.button(
                "Preparar Excel (multi-hoja)",
                key=f"{prefix}prepare_excel",
                width="stretch",
                help="Genera el libro trazable con datos, agregaciones, KPI y metadatos.",
            ):
                with st.spinner("Preparando el libro Excel…"):
                    st.session_state[excel_state_key] = dataframes_to_excel_bytes(sheets)
        if excel_state_key in st.session_state:
            st.download_button(
                "Descargar Excel (multi-hoja)",
                data=st.session_state[excel_state_key],
                file_name=timestamped_filename("heladas_puno_analisis", "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{prefix}download_excel",
                width="stretch",
            )
    with third:
        summary = build_report_summary(filtered, state)
        narrative = build_executive_summary(filtered, frost_column=state.frost_column)
        markdown = "# Resumen ejecutivo\n\n" + narrative + "\n\n"
        markdown += "\n".join(
            f"- **{row.Indicador}:** {row.Valor}" for row in summary.itertuples(index=False)
        )
        st.download_button(
            "Descargar resumen (.md)",
            data=markdown.encode("utf-8"),
            file_name=timestamped_filename("resumen_heladas_puno", "md"),
            mime="text/markdown",
            key=f"{prefix}download_md",
            width="stretch",
        )
