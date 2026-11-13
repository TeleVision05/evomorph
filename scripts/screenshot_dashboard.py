"""Serve the dashboard headless and screenshot every section with headless Chromium.

Usage (needs ``pip install playwright`` and its Chromium; nothing opens on screen):

    .venv/bin/python scripts/screenshot_dashboard.py --results results \
        --ga quick_ga_joint_s0 --me quick_me_joint_s0 [--out docs/media/dashboard]

Steps: start ``streamlit run`` with ``--server.headless true`` on a free local
port (no browser is opened), wait for its health check, fetch the page with a
plain HTTP request and print the status, then visit each section through the
``?run=...&section=...`` query parameters in headless Chromium, screenshot the
whole page, and stop the server. Each shot waits until Streamlit has finished
running the page (no spinner, no "running" indicator) so pictures are final.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "evomorph" / "visualization" / "dashboard.py"
WIDTH = 1500  # px; wide enough for the 6 headline tiles in one row
MAX_HEIGHT = 7000  # px; caps the full-page height of the longest section
SETTLE_MS = 2500  # extra wait after Streamlit is idle, for plotly to finish drawing
PAGE_TIMEOUT_MS = 300_000  # first visits render pictures and may take minutes


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(results: str, port: int, log: Path) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.headless", "true",
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",
        "--", "--results", results,
    ]  # fmt: skip
    env = {**os.environ, "MPLBACKEND": "Agg", "EVOMORPH_RESULTS": results}
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env)


def wait_healthy(port: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health") as r:
                if r.read().decode().strip() == "ok":
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("the dashboard did not start within 90 s")


def fetch_page(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
        body = r.read().decode()
        is_app = '<div id="root">' in body
        return f"HTTP {r.status}, {len(body)} bytes, streamlit page: {is_app}"


IDLE_JS = """() => {
  const running = document.querySelector('[data-testid="stStatusWidget"]');
  const spin = document.querySelector('[data-testid="stSpinner"]');
  const app = document.querySelector('[data-testid="stMain"]');
  return app && !spin && !(running && running.innerText.toLowerCase().includes('running'));
}"""


# The page scrolls inside an inner container: measure the main content block.
HEIGHT_JS = """() => {
  const b = document.querySelector('[data-testid="stMainBlockContainer"]');
  return b.getBoundingClientRect().top + b.scrollHeight + 60;
}"""


def shoot(page, url: str, out: Path, click: str | None = None) -> None:
    page.set_viewport_size({"width": WIDTH, "height": 1000})
    page.goto(url)
    page.wait_for_selector('[data-testid="stMain"]', timeout=PAGE_TIMEOUT_MS)
    page.wait_for_timeout(1500)
    page.wait_for_function(IDLE_JS, timeout=PAGE_TIMEOUT_MS)
    button = page.get_by_role("button", name=click) if click else None
    if button is not None and button.count():  # absent once the clip is cached
        button.first.click()
        page.wait_for_timeout(1500)
        page.wait_for_function(IDLE_JS, timeout=PAGE_TIMEOUT_MS)
    page.wait_for_timeout(SETTLE_MS)
    height = page.evaluate(HEIGHT_JS)
    page.set_viewport_size({"width": WIDTH, "height": min(int(height) + 40, MAX_HEIGHT)})
    page.wait_for_timeout(SETTLE_MS)
    page.screenshot(path=str(out))
    print(f"wrote {out}", flush=True)


def check_archive_click(page, port: int, run: str) -> bool:
    """Click the last occupied cell of the heat map; True if the shown elite changed.

    AppTest cannot click a plotly chart, so this is the check that a click on
    the chart (not only the picker below it) selects the cell.
    """
    page.set_viewport_size({"width": WIDTH, "height": 1400})
    page.goto(f"http://127.0.0.1:{port}/?run={run}&section=archive")
    page.wait_for_selector(".js-plotly-plot", timeout=PAGE_TIMEOUT_MS)
    page.wait_for_function(IDLE_JS, timeout=PAGE_TIMEOUT_MS)
    page.wait_for_timeout(SETTLE_MS)

    def shown() -> list[str]:
        text = page.locator('[data-testid="stMain"]').inner_text()
        return sorted(set(re.findall(r"cell \(\d+, \d+\) · \S+", text)))

    before = shown()
    markers = page.locator(".js-plotly-plot .scatterlayer .trace .points path")
    markers.nth(markers.count() - 1).click(force=True)
    page.wait_for_timeout(SETTLE_MS)
    page.wait_for_function(IDLE_JS, timeout=PAGE_TIMEOUT_MS)
    page.wait_for_timeout(SETTLE_MS)
    after = shown()
    print(f"archive click check ({markers.count()} cells): {before} -> {after}", flush=True)
    return bool(after) and after != before


def shots(args: argparse.Namespace) -> list[tuple[str, str, str, str | None]]:
    """(file name, run, section slug, button to click first or None)."""
    ga, me = args.ga, args.me
    extra = [("in_progress.png", args.in_progress, "evolution", None)] if args.in_progress else []
    return [
        *extra,
        ("evolution_viewer.png", ga, "evolution", None),
        ("creature_inspection.png", ga, "creature", "Render a clip"),
        ("map_elites_archive.png", me, "archive", None),
        ("statistics.png", ga, "stats", None),
        ("comparison.png", ga, "compare", "missing clip"),
        ("generation_history.png", me, "history", None),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--ga", required=True, help="GA run name for the GA sections")
    ap.add_argument("--me", required=True, help="MAP-Elites run name")
    ap.add_argument("--out", type=Path, default=Path("docs/media/dashboard"))
    ap.add_argument("--only", nargs="*", help="file names to (re)take; default all")
    ap.add_argument("--in-progress", help="also shoot this unfinished run as in_progress.png")
    ap.add_argument("--no-click-check", action="store_true", help="skip the heat-map click")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    log = Path(args.results) / ".dashboard_cache" / "server.log"  # gitignored, kept for debugging
    log.parent.mkdir(parents=True, exist_ok=True)
    server = start_server(args.results, port, log)
    try:
        wait_healthy(port)
        print(f"dashboard serving on 127.0.0.1:{port}: {fetch_page(port)}", flush=True)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(device_scale_factor=1)
            for name, run, section, click in shots(args):
                if args.only and name not in args.only:
                    continue
                url = f"http://127.0.0.1:{port}/?run={run}&section={section}"
                shoot(page, url, args.out / name, click)
            if not args.no_click_check:
                clicked = check_archive_click(page, port, args.me)
                print(f"clicking a heat-map cell selects it: {clicked}", flush=True)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)
        print(f"server stopped (exit code {server.returncode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
