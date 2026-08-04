"""Genera capturas reproducibles del dashboard mediante Playwright."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "capturas"

PAGES: tuple[tuple[str, str, str | None], ...] = (
    ("01_vista_general.png", "Vista general", None),
    ("02_analisis_descriptivo.png", "Análisis descriptivo", "Analisis descriptivo"),
    ("03_analisis_multidimensional.png", "Análisis multidimensional", "Analisis multidimensional"),
    ("04_analisis_geografico.png", "Análisis geográfico", "Analisis geografico"),
    ("05_conclusiones.png", "Conclusiones", "Conclusiones"),
    ("06_metodologia_datos.png", "Metodología y datos", "Metodologia y datos"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument(
        "--solo",
        help="Genera sólo una captura (nombre de archivo o fragmento del título).",
    )
    parser.add_argument(
        "--sin-servidor",
        action="store_true",
        help="Usa un servidor ya levantado en el puerto indicado.",
    )
    return parser.parse_args()


def wait_for_port(port: int, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Streamlit no respondió en el puerto {port} tras {timeout} s.")


def start_server(port: int) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.runOnSave",
        "false",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        creationflags=flags,
    )


def settle(page: Page, timeout_ms: int = 180_000) -> None:
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=timeout_ms)
    page.wait_for_load_state("networkidle", timeout=timeout_ms)
    # Streamlit puede completar la navegación antes que los modelos ejecutados
    # dentro del script. Se deja aparecer el spinner y, si existe, se espera a
    # que desaparezca para no capturar una página analítica a medio construir.
    page.wait_for_timeout(3_000)
    spinner = page.locator('[data-testid="stSpinner"]')
    if spinner.count():
        spinner.first.wait_for(state="hidden", timeout=timeout_ms)
    page.wait_for_timeout(4_000)


def navigate(page: Page, label: str) -> None:
    link = page.get_by_role("link", name=label, exact=False).first
    link.scroll_into_view_if_needed()
    link.click()
    settle(page)


def capture(page: Page, filename: str) -> None:
    page.screenshot(path=str(OUTPUT / filename), full_page=True, animations="disabled")
    size = (OUTPUT / filename).stat().st_size
    if size < 20_000:
        raise RuntimeError(f"La captura {filename} parece vacía ({size} bytes).")
    print(f"  ✓ {filename} ({size / 1024:.0f} KiB)")


def main() -> int:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    server: subprocess.Popen[str] | None = None
    try:
        if not args.sin_servidor:
            print(f"Levantando Streamlit en http://127.0.0.1:{args.port}…")
            server = start_server(args.port)
        wait_for_port(args.port)

        requested = PAGES
        if args.solo:
            needle = args.solo.casefold()
            requested = tuple(
                item for item in PAGES if needle in item[0].casefold() or needle in item[1].casefold()
            )
            if not requested:
                raise ValueError(f"No se encontró una captura que coincida con {args.solo!r}.")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1680, "height": 1050},
                device_scale_factor=2,
                locale="es-PE",
                color_scheme="light",
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{args.port}", wait_until="domcontentloaded", timeout=180_000)
            settle(page)

            for filename, _title, navigation_label in requested:
                if navigation_label:
                    navigate(page, navigation_label)
                elif page.url.rstrip("/").endswith(str(args.port)) is False:
                    page.goto(f"http://127.0.0.1:{args.port}", timeout=180_000)
                    settle(page)
                capture(page, filename)

            # Los recortes complementarios se generan cuando se captura la vista general.
            if not args.solo or any("vista_general" in item[0] for item in requested):
                page.goto(f"http://127.0.0.1:{args.port}", timeout=180_000)
                settle(page)
                sidebar = page.locator('[data-testid="stSidebar"]').first
                sidebar.screenshot(path=str(OUTPUT / "07_barra_lateral_filtros.png"), animations="disabled")
                kpis = page.locator('[data-testid="stHorizontalBlock"]').filter(
                    has=page.locator(".pn-kpi")
                ).first
                kpis.screenshot(path=str(OUTPUT / "08_indicadores_kpi.png"), animations="disabled")
                print("  ✓ 07_barra_lateral_filtros.png")
                print("  ✓ 08_indicadores_kpi.png")

            context.close()
            browser.close()
        print(f"Capturas guardadas en {OUTPUT}")
        return 0
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
