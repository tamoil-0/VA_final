"""Genera cifras y figuras reproducibles para el informe tecnico.

El script lee exclusivamente el dataset analitico del proyecto y reutiliza las
funciones de dominio empleadas por el dashboard.  De ese modo las cifras del
PDF no dependen de transcripciones manuales y pueden auditarse con una sola
ejecucion::

    python report/generar_figuras_informe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import MULTIVARIATE_FEATURES  # noqa: E402
from utils.metrics import (  # noqa: E402
    aggregate_annual,
    aggregate_by_province,
    aggregate_monthly,
    compare_sources,
    frost_calendar,
)
from utils.ml_models import (  # noqa: E402
    detect_anomalies,
    run_kmeans,
    run_pca,
    train_frost_classifier,
)
from utils.stats_tools import fit_linear_trend, mann_kendall_test  # noqa: E402

BLUE = "#578BC5"
ORANGE = "#DB8963"
INK = "#263445"
GRID = "#DCE3EC"
SURFACE = "#F7F8FA"


def _style() -> None:
    """Aplica una identidad visual coherente con el dashboard."""

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    """Guarda una figura con margenes ajustados y resolucion de impresion."""

    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _native(value):
    """Convierte objetos NumPy/Pandas en tipos serializables por JSON."""

    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, (pd.Series, pd.Index)):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _latex_escape(value: object) -> str:
    """Escapa texto procedente de CSV/JSON para incluirlo en LaTeX."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_\allowbreak{}",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "≤": r"\ensuremath{\leq}",
        "≥": r"\ensuremath{\geq}",
        "→": r"\ensuremath{\rightarrow}",
        "×": r"\ensuremath{\times}",
        "≈": r"\ensuremath{\approx}",
        "°": r"\textdegree{}",
        "²": r"\textsuperscript{2}",
        "³": r"\textsuperscript{3}",
        "₀": r"\textsubscript{0}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _decimal(value: float, digits: int = 2) -> str:
    """Formatea un decimal para una macro LaTeX usando coma española."""

    return f"{value:.{digits}f}".replace(".", ",")


def _write_latex_support(frame: pd.DataFrame, results: dict) -> None:
    """Genera macros, diccionario completo y bitacora para los anexos."""

    report_dir = ROOT / "report"
    summary = results["resumen"]
    trend = results["tendencia"]
    territory = results["territorio"]
    validation = results["validacion_cruzada"]
    pca_results = results["pca"]
    kmeans_results = results["kmeans"]
    anomaly_results = results["anomalias"]
    forest = results["random_forest"]
    july = next(item for item in results["meses"] if int(item["mes"]) == 7)

    macros = rf"""% Archivo generado. No editar a mano.
\newcommand{{\NRegistros}}{{47\,489}}
\newcommand{{\NVariables}}{{{results['estructura']['variables']}}}
\newcommand{{\NProvincias}}{{{results['estructura']['provincias']}}}
\newcommand{{\NDias}}{{3\,653}}
\newcommand{{\NHeladas}}{{{summary['heladas_meteorologicas']:,}}}
\newcommand{{\NHeladasAgronomicas}}{{{summary['heladas_agronomicas']:,}}}
\newcommand{{\TasaHelada}}{{{_decimal(summary['tasa_helada_meteorologica_pct'], 2)}\,\%}}
\newcommand{{\TasaHeladaAgronomica}}{{{_decimal(summary['tasa_helada_agronomica_pct'], 2)}\,\%}}
\newcommand{{\FactorAgronomico}}{{{_decimal(summary['factor_umbral_agronomico'], 2)}}}
\newcommand{{\TminMedia}}{{{_decimal(summary['temperatura_minima_media_c'], 2)}~\textdegree C}}
\newcommand{{\TminAbsoluta}}{{{_decimal(summary['temperatura_minima_absoluta_c'], 1)}~\textdegree C}}
\newcommand{{\RiesgoMedio}}{{{_decimal(summary['indice_riesgo_medio'], 2)}}}
\newcommand{{\TasaJulio}}{{{_decimal(july['tasa_helada'], 2)}\,\%}}
\newcommand{{\OLSPendiente}}{{{_decimal(trend['ols_pendiente_dias_por_anio'], 2)}}}
\newcommand{{\OLSRdos}}{{{_decimal(trend['ols_r2'], 3)}}}
\newcommand{{\OLSp}}{{{_decimal(trend['ols_p'], 3)}}}
\newcommand{{\MKTau}}{{{_decimal(trend['mann_kendall_tau'], 3)}}}
\newcommand{{\MKp}}{{{_decimal(trend['mann_kendall_p'], 3)}}}
\newcommand{{\SenPendiente}}{{{_decimal(trend['sen_dias_por_anio'], 2)}}}
\newcommand{{\AltitudR}}{{{_decimal(territory['altitud_tasa_r'], 3)}}}
\newcommand{{\AltitudP}}{{{_decimal(territory['altitud_tasa_p'], 3)}}}
\newcommand{{\LagoTasa}}{{{_decimal(territory['circunlacustre_tasa_pct'], 2)}\,\%}}
\newcommand{{\RestoTasa}}{{{_decimal(territory['resto_tasa_pct'], 2)}\,\%}}
\newcommand{{\FuenteR}}{{{_decimal(validation['correlacion'], 3)}}}
\newcommand{{\FuenteSesgo}}{{{_decimal(validation['sesgo_medio'], 2)}~\textdegree C}}
\newcommand{{\FuenteMAE}}{{{_decimal(validation['error_absoluto_medio'], 2)}~\textdegree C}}
\newcommand{{\FuenteConcordancia}}{{{_decimal(validation['concordancia_helada'], 2)}\,\%}}
\newcommand{{\PCAUno}}{{{_decimal(pca_results['varianza_explicada_pct'][0], 2)}\,\%}}
\newcommand{{\PCADosAcum}}{{{_decimal(pca_results['varianza_acumulada_pct'][1], 2)}\,\%}}
\newcommand{{\PCANoventa}}{{{pca_results['componentes_90_pct']}}}
\newcommand{{\KOptimo}}{{{kmeans_results['k']}}}
\newcommand{{\Silueta}}{{{_decimal(kmeans_results['silueta'], 3)}}}
\newcommand{{\NAnomalias}}{{{anomaly_results['cantidad']}}}
\newcommand{{\RFAUC}}{{{_decimal(forest['roc_auc'], 3)}}}
\newcommand{{\RFAP}}{{{_decimal(forest['precision_media'], 3)}}}
\newcommand{{\RFExactitud}}{{{_decimal(100 * forest['exactitud'], 2)}\,\%}}
\newcommand{{\RFSensibilidad}}{{{_decimal(100 * forest['sensibilidad'], 2)}\,\%}}
\newcommand{{\RFPrecision}}{{{_decimal(100 * forest['precision'], 2)}\,\%}}
""".replace("7,620", "7\\,620").replace("18,877", "18\\,877")
    (report_dir / "resultados_macros.tex").write_text(macros, encoding="utf-8")

    dictionary = pd.read_csv(ROOT / "data" / "diccionario_datos.csv")
    rows = [
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{2.8cm}"
        r">{\raggedright\arraybackslash}p{2.9cm}"
        r">{\centering\arraybackslash}p{1.2cm}"
        r">{\centering\arraybackslash}p{1.5cm}"
        r">{\raggedright\arraybackslash}p{5.5cm}}",
        r"\caption{Diccionario completo del dataset analítico}\label{tab:diccionario-completo}\\",
        r"\toprule",
        r"\textbf{Variable} & \textbf{Etiqueta} & \textbf{Unidad} & \textbf{Origen} & \textbf{Descripción} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\small\itshape Continuación de la tabla~\ref{tab:diccionario-completo}}\\",
        r"\toprule",
        r"\textbf{Variable} & \textbf{Etiqueta} & \textbf{Unidad} & \textbf{Origen} & \textbf{Descripción} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{5}{r}{\small Continúa en la página siguiente}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for record in dictionary.to_dict("records"):
        rows.append(
            " & ".join(
                _latex_escape(record[column])
                for column in ("variable", "etiqueta", "unidad", "procedencia", "descripcion")
            )
            + r" \\"
        )
    rows.append(r"\end{longtable}")
    (report_dir / "diccionario_variables.tex").write_text("\n".join(rows), encoding="utf-8")

    metadata = json.loads((ROOT / "data" / "metadata.json").read_text(encoding="utf-8"))
    steps = metadata["pipeline_etl"]["steps"]
    trace = [r"\begin{enumerate}[leftmargin=*,itemsep=0.35em]"]
    trace.extend(r"\item " + _latex_escape(step) for step in steps)
    trace.append(r"\end{enumerate}")
    (report_dir / "bitacora_etl.tex").write_text("\n".join(trace), encoding="utf-8")


def main() -> int:
    """Calcula resultados, crea las figuras y persiste la trazabilidad."""

    _style()
    output = ROOT / "report" / "figuras"
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(ROOT / "data" / "datos_limpios.csv", parse_dates=["fecha"])
    province = aggregate_by_province(frame)
    monthly = aggregate_monthly(frame)
    annual = aggregate_annual(frame)
    trend = fit_linear_trend(annual["anio"], annual["heladas_por_provincia"])
    mann_kendall = mann_kendall_test(annual["heladas_por_provincia"])
    pca = run_pca(frame, list(MULTIVARIATE_FEATURES), n_components=len(MULTIVARIATE_FEATURES))
    clusters = run_kmeans(frame, list(MULTIVARIATE_FEATURES), max_clusters=4)
    anomalies = detect_anomalies(frame, list(MULTIVARIATE_FEATURES), contamination=0.01, top=15)
    classifier = train_frost_classifier(
        frame, list(MULTIVARIATE_FEATURES), target="helada", test_size=0.25
    )

    if any(item is None for item in (trend, mann_kendall, pca, clusters, anomalies, classifier)):
        raise RuntimeError("Una tecnica analitica no produjo resultados con el dataset completo.")

    coldest = frame.loc[frame["temperatura_minima"].idxmin()]
    altitude_fit = stats.linregress(province["altitud"], province["tasa_helada"])
    lake = frame["zona_agroecologica"].eq("Altiplano circunlacustre")
    source = compare_sources(frame)

    results = {
        "generado_desde": "data/datos_limpios.csv",
        "semilla": 42,
        "estructura": {
            "registros": len(frame),
            "variables": len(frame.columns),
            "provincias": frame["provincia"].nunique(),
            "dias": frame["fecha"].nunique(),
            "inicio": frame["fecha"].min(),
            "fin": frame["fecha"].max(),
        },
        "resumen": {
            "heladas_meteorologicas": int(frame["helada"].sum()),
            "tasa_helada_meteorologica_pct": 100 * frame["helada"].mean(),
            "heladas_agronomicas": int(frame["helada_agronomica"].sum()),
            "tasa_helada_agronomica_pct": 100 * frame["helada_agronomica"].mean(),
            "factor_umbral_agronomico": frame["helada_agronomica"].mean() / frame["helada"].mean(),
            "temperatura_minima_media_c": frame["temperatura_minima"].mean(),
            "temperatura_minima_absoluta_c": frame["temperatura_minima"].min(),
            "minima_absoluta_provincia": coldest["provincia"],
            "minima_absoluta_fecha": coldest["fecha"],
            "indice_riesgo_medio": frame["indice_riesgo_helada"].mean(),
            "indice_riesgo_p95": frame["indice_riesgo_helada"].quantile(0.95),
        },
        "intensidad": frame["intensidad_helada"].value_counts().to_dict(),
        "niveles_riesgo": frame["nivel_riesgo"].value_counts().to_dict(),
        "provincias": province[
            [
                "provincia",
                "dias_helada",
                "tasa_helada",
                "tmin_media",
                "tmin_absoluta",
                "riesgo_medio",
                "altitud",
            ]
        ].to_dict("records"),
        "meses": monthly[
            ["mes", "mes_abrev", "dias_helada", "tasa_helada", "tmin_media", "precipitacion_media"]
        ].to_dict("records"),
        "anios": annual[
            ["anio", "heladas_por_provincia", "tasa_helada", "tmin_media", "tmin_absoluta"]
        ].to_dict("records"),
        "tendencia": {
            "ols_pendiente_dias_por_anio": trend.slope,
            "ols_r2": trend.r_squared,
            "ols_p": trend.p_value,
            "ols_ic95": trend.confidence_interval,
            "mann_kendall_s": mann_kendall.statistic_s,
            "mann_kendall_tau": mann_kendall.tau,
            "mann_kendall_p": mann_kendall.p_value,
            "sen_dias_por_anio": mann_kendall.sen_slope,
        },
        "territorio": {
            "altitud_tasa_r": altitude_fit.rvalue,
            "altitud_tasa_p": altitude_fit.pvalue,
            "pendiente_tasa_por_100m": altitude_fit.slope * 100,
            "circunlacustre_tasa_pct": 100 * frame.loc[lake, "helada"].mean(),
            "circunlacustre_tmin_c": frame.loc[lake, "temperatura_minima"].mean(),
            "resto_tasa_pct": 100 * frame.loc[~lake, "helada"].mean(),
            "resto_tmin_c": frame.loc[~lake, "temperatura_minima"].mean(),
        },
        "correlaciones_con_tmin": {
            variable: {
                "pearson": frame[variable].corr(frame["temperatura_minima"]),
                "spearman": frame[variable].corr(frame["temperatura_minima"], method="spearman"),
            }
            for variable in [
                "altitud",
                "humedad_relativa",
                "nubosidad",
                "radiacion_solar",
                "precipitacion",
                "velocidad_viento",
                "temperatura_suelo",
            ]
        },
        "validacion_cruzada": source,
        "pca": {
            "varianza_explicada_pct": 100 * pca.explained_variance,
            "varianza_acumulada_pct": 100 * pca.cumulative_variance,
            "componentes_90_pct": pca.n_components_90,
            "variables_dominantes": {
                f"CP{index + 1}": pca.dominant_variables(index, top=4) for index in range(3)
            },
        },
        "kmeans": {
            "k": clusters.n_clusters,
            "silueta": clusters.silhouette,
            "calidad": clusters.quality,
            "diagnostico": clusters.diagnostics.to_dict("records"),
            "perfiles": clusters.profile.to_dict("records"),
        },
        "anomalias": {
            "cantidad": anomalies.n_anomalies,
            "porcentaje": 100 * anomalies.n_anomalies / len(frame),
            "extremos": anomalies.extremes.to_dict("records"),
        },
        "random_forest": {
            "roc_auc": classifier.roc_auc,
            "precision_media": classifier.average_precision,
            "exactitud": classifier.accuracy,
            "sensibilidad": classifier.recall,
            "precision": classifier.precision,
            "entrenamiento": classifier.n_train,
            "prueba": classifier.n_test,
            "matriz_confusion": classifier.confusion.tolist(),
            "importancias": classifier.importances.to_dict("records"),
        },
    }
    (ROOT / "report" / "resultados.json").write_text(
        json.dumps(_native(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_latex_support(frame, results)

    # Figura 1: ranking territorial.
    ordered = province.sort_values("dias_helada")
    colors = [ORANGE if name == "Carabaya" else BLUE for name in ordered["provincia"]]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    bars = ax.barh(ordered["provincia"], ordered["dias_helada"], color=colors)
    ax.bar_label(bars, padding=3, fmt="%.0f", fontsize=8)
    ax.set_title("Días con helada meteorológica por provincia, 2015–2024")
    ax.set_xlabel("Días con temperatura mínima ≤ 0 °C")
    ax.set_ylabel("")
    ax.spines[["top", "right", "left"]].set_visible(False)
    _save(fig, output / "01_ranking_provincial.png")

    # Figura 2: ciclo anual en dos paneles, sin doble eje.
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.7), sharex=True)
    axes[0].plot(monthly["mes_abrev"], monthly["tasa_helada"], color=ORANGE, linewidth=2.2, marker="o")
    axes[0].fill_between(monthly["mes_abrev"], monthly["tasa_helada"], color=ORANGE, alpha=0.16)
    axes[0].set_ylabel("Tasa de helada (%)")
    axes[0].set_title("Ciclo estacional: heladas y precipitación diaria media")
    axes[1].bar(monthly["mes_abrev"], monthly["precipitacion_media"], color=BLUE)
    axes[1].set_ylabel("Precipitación (mm/día)")
    axes[1].set_xlabel("Mes")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output / "02_ciclo_anual.png")

    # Figura 3: serie anual y tendencia OLS.
    years = annual["anio"].to_numpy(dtype=float)
    fitted = trend.predict(years)
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.plot(years, annual["heladas_por_provincia"], color=BLUE, linewidth=2.2, marker="o", label="Observado")
    ax.plot(years, fitted, color=ORANGE, linewidth=2.0, linestyle="--", label="OLS")
    ax.set_title("Evolución anual de las heladas por provincia")
    ax.set_xlabel("Año")
    ax.set_ylabel("Días con helada por provincia")
    ax.set_xticks(annual["anio"])
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output / "03_tendencia_anual.png")

    # Figura 4: correlaciones relevantes.
    correlation_variables = [
        "temperatura_minima",
        "temperatura_suelo",
        "altitud",
        "humedad_relativa",
        "nubosidad",
        "precipitacion",
        "radiacion_solar",
        "velocidad_viento",
    ]
    labels = ["T. mínima", "T. suelo", "Altitud", "Humedad", "Nubosidad", "Precipitación", "Radiación", "Viento"]
    correlation = frame[correlation_variables].corr()
    fig, ax = plt.subplots(figsize=(7.4, 6.1))
    sns.heatmap(
        correlation,
        cmap=sns.diverging_palette(235, 25, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "r de Pearson", "shrink": 0.78},
        ax=ax,
    )
    ax.set_title("Matriz de correlación de variables agroclimáticas")
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, output / "04_correlaciones.png")

    # Figura 5: varianza de PCA y proyección muestral CP1–CP2.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    component_numbers = np.arange(1, len(pca.explained_variance) + 1)
    axes[0].bar(component_numbers, 100 * pca.explained_variance, color=BLUE)
    axes[0].axhline(10, color=ORANGE, linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Componente principal")
    axes[0].set_ylabel("Varianza explicada (%)")
    axes[0].set_title("Aporte individual")
    axes[1].plot(component_numbers, 100 * pca.cumulative_variance, color=ORANGE, marker="o", linewidth=2)
    axes[1].axhline(90, color=INK, linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Número de componentes")
    axes[1].set_ylabel("Varianza acumulada (%)")
    axes[1].set_ylim(0, 103)
    axes[1].set_title("Aporte acumulado")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Reducción de dimensionalidad mediante PCA", fontsize=12, fontweight="bold")
    _save(fig, output / "05_pca_varianza.png")

    # Figura 6: concordancia entre las fuentes independientes.
    paired = frame[["temperatura_minima", "temperatura_minima_merra2"]].dropna()
    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    density = ax.hexbin(
        paired["temperatura_minima_merra2"],
        paired["temperatura_minima"],
        gridsize=48,
        mincnt=1,
        cmap="Blues",
    )
    limits = [
        min(paired.min().min(), -12),
        max(paired.max().max(), 20),
    ]
    ax.plot(limits, limits, color=ORANGE, linestyle="--", linewidth=1.8, label="Concordancia perfecta")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Temperatura mínima MERRA-2 (°C)")
    ax.set_ylabel("Temperatura mínima ERA5-Land (°C)")
    ax.set_title("Validación cruzada entre fuentes")
    ax.legend(frameon=False)
    fig.colorbar(density, ax=ax, label="Número de observaciones")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output / "06_validacion_fuentes.png")

    # Figura 7: calendario territorial de heladas.
    calendar = frost_calendar(frame)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.heatmap(
        calendar,
        cmap=sns.light_palette(BLUE, as_cmap=True),
        annot=True,
        fmt=".0f",
        linewidths=0.35,
        cbar_kws={"label": "Tasa de helada (%)", "shrink": 0.78},
        ax=ax,
    )
    ax.set_title("Calendario territorial de incidencia de heladas")
    ax.set_xlabel("Mes")
    ax.set_ylabel("")
    _save(fig, output / "07_calendario_heladas.png")

    print(f"Resultados: {ROOT / 'report' / 'resultados.json'}")
    print(f"Figuras: {len(list(output.glob('*.png')))} archivos en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
