# SPDX-License-Identifier: Apache-2.0
"""Screenshots of the five pages in dark and light at phone, tablet and desktop widths, from the fixture server.

    uv run --with playwright python ui/tests/shoot.py OUT_DIR

Needs a Chromium for Playwright (`playwright install chromium`).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
import fixture_server

PAGES = [
    ("health", "#/health"),
    ("queries", "#/queries"),
    ("query-detail", "#/queries/01932a40-1a00-7a01-8001-aaaaaaaaaa02"),
    ("history", "#/history"),
    ("resources", "#/resources"),
]


VIEWPORTS = [("phone", 390, 844), ("tablet", 820, 1180), ("desktop", 1440, 900)]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    server = fixture_server.serve()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        overflow = []
        for scheme in ("dark", "light"):
            for label, width, height in VIEWPORTS:
                ctx = browser.new_context(
                    color_scheme=scheme,
                    viewport={"width": width, "height": height},
                    has_touch=label != "desktop",
                    is_mobile=label == "phone",
                )
                page = ctx.new_page()
                page.goto(f"{base}/?token=fixture")
                for name, frag in PAGES:
                    page.evaluate(f"window.location.hash = {frag!r}")
                    page.wait_for_timeout(1200)
                    page.screenshot(
                        path=str(out / f"{name}-{label}-{scheme}.png"), full_page=True
                    )
                    wide = page.evaluate(
                        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
                    )
                    if wide > 0:
                        overflow.append(
                            f"{name} {label} {scheme}: page is {wide}px wider than the screen"
                        )
                ctx.close()
        (out / "overflow.txt").write_text(
            "\n".join(overflow) + ("\n" if overflow else "")
        )
        print("horizontal page scroll:", len(overflow), "of", 5 * 3 * 2, "views")
        browser.close()
    print("wrote", len(list(out.glob("*.png"))), "screenshots to", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
