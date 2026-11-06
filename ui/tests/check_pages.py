# SPDX-License-Identifier: Apache-2.0
"""Snapshot check of the five UI pages at phone, tablet and desktop widths, dark and light.

    uv run --with playwright python ui/tests/check_pages.py [SHOT_DIR]

Serves the built UI from the fixture server and fails when a page: scrolls sideways, loads anything from another
origin, logs a console error, lacks the self-hosted Red Hat fonts, loses its key text, or has a navigation target
under 40 px on a phone. With SHOT_DIR it also writes a PNG per page and view (the visual record; the assertions
are the check, since pixels differ between machines).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
import fixture_server

VIEWPORTS = [("phone", 390, 844), ("tablet", 820, 1180), ("desktop", 1440, 900)]
PAGES = [
    ("health", "#/health", ["Health", "Caches", "hit rate", "Uptime"]),
    ("queries", "#/queries", ["Queries", "SELECT l_returnflag", "Cancel"]),
    (
        "query-detail",
        "#/queries/01932a40-1a00-7a01-8001-aaaaaaaaaa02",
        ["HashJoin", "128.0 MiB", "Operators"],
    ),
    ("history", "#/history", ["History", "DIVIDE_BY_ZERO", "Compare the two selected"]),
    ("resources", "#/resources", ["Memory", "Caches", "root/query-1"]),
]
SMALL_TARGETS = """
() => [...document.querySelectorAll('button, a, input, select')]
  .filter(e => e.offsetParent !== null)
  .map(e => [e, (e.tagName === 'INPUT' && e.closest('label') ? e.closest('label') : e).getBoundingClientRect()])
  .filter(([e, r]) => (r.height < 40 || r.width < 40) && !(e.tagName === 'A' && e.closest('main') && e.closest('p, td')))
  .map(([e, r]) => `${e.tagName.toLowerCase()} "${(e.textContent || e.getAttribute('aria-label') || '').trim().slice(0, 30)}" ${Math.round(r.width)}x${Math.round(r.height)}`)
"""


def check(shots: Path | None) -> list[str]:
    problems: list[str] = []
    server = fixture_server.serve()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for scheme in ("dark", "light"):
            for label, width, height in VIEWPORTS:
                ctx = browser.new_context(
                    color_scheme=scheme,
                    viewport={"width": width, "height": height},
                    has_touch=label != "desktop",
                    is_mobile=label == "phone",
                )
                page = ctx.new_page()
                errors: list[str] = []
                foreign: list[str] = []
                page.on(
                    "console",
                    lambda m, errors=errors: (
                        errors.append(m.text) if m.type == "error" else None
                    ),
                )
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                page.on(
                    "request",
                    lambda r, foreign=foreign: (
                        foreign.append(r.url)
                        if not r.url.startswith((origin, "data:", "blob:"))
                        else None
                    ),
                )
                page.goto(f"{origin}/?token=fixture")
                for name, frag, texts in PAGES:
                    view = f"{name} {label} {scheme}"
                    page.evaluate(f"window.location.hash = {frag!r}")
                    page.wait_for_timeout(1000)
                    wide = page.evaluate(
                        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
                    )
                    if wide > 0:
                        problems.append(
                            f"{view}: the page scrolls sideways ({wide}px too wide)"
                        )
                    body = page.inner_text("body")
                    problems += [
                        f"{view}: missing text {t!r}"
                        for t in texts
                        if t.lower() not in body.lower()
                    ]
                    if label == "phone":
                        small = page.evaluate(SMALL_TARGETS)
                        problems += [
                            f"{view}: touch target under 40 px: {s}" for s in small
                        ]
                    if shots is not None:
                        shots.mkdir(parents=True, exist_ok=True)
                        page.screenshot(
                            path=str(shots / f"{name}-{label}-{scheme}.png"),
                            full_page=True,
                        )
                fonts = page.evaluate(
                    "Promise.all([document.fonts.load('14px \"Red Hat Text\"'),"
                    " document.fonts.load('700 24px \"Red Hat Display\"'),"
                    " document.fonts.load('13px \"Red Hat Mono\"')]).then(r => r.map(f => f.length))"
                )
                if not all(fonts):
                    problems.append(
                        f"{label} {scheme}: a Red Hat font did not load: {fonts}"
                    )
                problems += [f"{label} {scheme}: console error: {e}" for e in errors]
                problems += [
                    f"{label} {scheme}: request outside the page's origin: {u}"
                    for u in foreign
                ]
                if label == "phone":
                    page.evaluate("window.location.hash = '#/health'")
                    page.wait_for_timeout(200)
                    if page.get_by_role("dialog").count() != 0:
                        problems.append(
                            f"phone {scheme}: the menu is open before it is asked for"
                        )
                    page.get_by_role("button", name="Open the menu").click()
                    page.wait_for_timeout(300)
                    dialog = page.get_by_role("dialog")
                    if (
                        dialog.count() != 1
                        or not dialog.get_by_role("link", name="Queries").is_visible()
                    ):
                        problems.append(
                            f"phone {scheme}: the menu button does not open the pages"
                        )
                    else:
                        dialog.get_by_role("link", name="Queries").click()
                        page.wait_for_timeout(300)
                        if (
                            page.get_by_role("dialog").count() != 0
                            or "#/queries" not in page.url
                        ):
                            problems.append(
                                f"phone {scheme}: choosing a page does not close the menu and go there"
                            )
                ctx.close()
        browser.close()
    return problems


if __name__ == "__main__":
    found = check(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(
        "\n".join(found) if found else "ui pages: ok (5 pages x 3 widths x dark/light)"
    )
    sys.exit(1 if found else 0)
