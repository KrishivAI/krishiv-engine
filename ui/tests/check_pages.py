# SPDX-License-Identifier: Apache-2.0
"""Snapshot check of the eight UI pages at phone, tablet and desktop widths, dark and light.

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

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
import fixture_server

# "small" is the narrowest width the spec names (360 px); it is checked but not photographed.
VIEWPORTS = [
    ("small", 360, 740),
    ("phone", 390, 844),
    ("tablet", 820, 1180),
    ("desktop", 1440, 900),
]
PHONES = ("small", "phone")
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
    ("catalog", "#/catalog", ["Catalog", "spark_catalog", "Schemas", "Tables"]),
    ("sql", "#/sql", ["SQL", "Run", "Cancel"]),
    ("environment", "#/environment", ["Environment", "krishiv.auth.token", "***"]),
]
# The four pages the axe check covers (the spec's "four pages": queries, catalog, SQL editor, environment).
AXE_PAGES = ["#/queries", "#/catalog", "#/sql", "#/environment"]
AXE = Path(__file__).parent.parent / "node_modules" / "axe-core" / "axe.min.js"
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
                    is_mobile=label in PHONES,
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
                    if label in PHONES:
                        small = page.evaluate(SMALL_TARGETS)
                        problems += [
                            f"{view}: touch target under 40 px: {s}" for s in small
                        ]
                    if shots is not None and label != "small":
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
                if label in PHONES:
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
                # Collected last, so an error the menu (or anything above) raises is reported too.
                problems += [f"{label} {scheme}: console error: {e}" for e in errors]
                problems += [
                    f"{label} {scheme}: request outside the page's origin: {u}"
                    for u in foreign
                ]
                ctx.close()
        browser.close()
    return problems


def page_checks(origin: str) -> list[str]:
    """XSS, session, cancel and accessibility checks on the desktop dark view of the fixture server."""
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on(
            "dialog", lambda d: (errors.append(f"dialog: {d.message}"), d.dismiss())
        )
        page.goto(f"{origin}/?token=fixture")
        # Ready: the shell shows its navigation and the token has left the address.
        page.wait_for_function("!location.search.includes('token')", timeout=10000)
        page.get_by_role("navigation").first.wait_for(timeout=10000)
        # Session: the token left the address and no script can read it.
        if "token" in page.url:
            problems.append(f"the token is still in the address: {page.url}")
        cookies = [c for c in ctx.cookies() if c["name"] == "krishiv_ui_token"]
        if len(cookies) != 1:
            problems.append(f"expected one krishiv_ui_token cookie, got {cookies}")
        else:
            c = cookies[0]
            if not c["httpOnly"]:
                problems.append("the session cookie is not HttpOnly")
            if c["sameSite"] != "Strict":
                problems.append(f"the session cookie is SameSite={c['sameSite']}")
            if c["path"] != "/api/internal":
                problems.append(f"the session cookie path is {c['path']}")
        if page.evaluate("document.cookie").find("fixture") >= 0:
            problems.append("the session cookie is readable by script (not HttpOnly)")
        if (
            page.evaluate("JSON.stringify([sessionStorage, localStorage])").find(
                "fixture"
            )
            >= 0
        ):
            problems.append("the token is kept in web storage")
        # XSS: hostile text in query text, principals, catalog names, comments, results and errors renders as text.

        def no_markup(view: str) -> None:
            if page.evaluate("window.__xss") is not None:
                problems.append(f"{view}: hostile text ran as script")
            if page.locator('img[src="x"]').count() != 0:
                problems.append(f"{view}: hostile text was rendered as an element")

        page.evaluate("window.location.hash = '#/queries'")
        page.get_by_text("<img src=x", exact=False).first.wait_for(timeout=10000)
        no_markup("queries")
        page.evaluate("window.location.hash = '#/catalog'")
        page.get_by_role("button", name="spark_catalog").wait_for(timeout=10000)
        no_markup("catalog")
        page.get_by_role("button", name="spark_catalog").click()
        page.get_by_role("button", name="default").click()
        page.get_by_role("button", name="orders").click()
        page.get_by_text("o_orderkey").wait_for(timeout=10000)
        no_markup("catalog columns")
        page.evaluate("window.location.hash = '#/sql'")
        editor = page.locator(".cm-content")
        editor.wait_for(timeout=10000)
        for text, outcome in (("SELECT 1", "finished"), ("boom", "failed")):
            editor.click()
            page.keyboard.press("Control+A")
            page.keyboard.type(text)
            page.get_by_role("button", name="Run").click()
            page.get_by_text(outcome, exact=True).first.wait_for(timeout=10000)
            no_markup(f"sql {text}")
        # Cancel shows the cancelled state within 2 s.
        editor.click()
        page.keyboard.press("Control+A")
        page.keyboard.type("sleep")
        page.get_by_role("button", name="Run").click()
        page.get_by_text("running", exact=True).first.wait_for(timeout=10000)
        page.get_by_role("button", name="Cancel").click()
        try:
            page.get_by_text("cancelled", exact=True).wait_for(timeout=2000)
        except PlaywrightTimeout:
            problems.append(
                "the SQL editor did not show the cancelled state within 2 s"
            )
        # Redaction on the environment page.
        page.evaluate("window.location.hash = '#/environment'")
        page.get_by_text("spark.sql.shuffle.partitions", exact=True).first.wait_for(
            timeout=10000
        )
        for key in (
            "krishiv.auth.token",
            "db.password",
            "spark.hadoop.fs.s3a.secret.key",
        ):
            row = page.locator("tr", has=page.get_by_text(key, exact=True)).first
            shown = row.locator("span.mono").nth(1).inner_text().strip()
            if shown != "***":
                problems.append(f"environment: {key} shows {shown!r}, not '***'")
        # Accessibility: no serious or critical axe violation on the four pages.
        if not AXE.exists():
            problems.append(
                f"axe-core is not installed ({AXE}); run pnpm install in ui/"
            )
        else:
            axe_js = AXE.read_text()
            for frag in AXE_PAGES:
                page.evaluate(f"window.location.hash = {frag!r}")
                # The event stream keeps the network busy, so "networkidle" never comes: wait for the page.
                page.locator("h1").first.wait_for(timeout=10000)
                page.wait_for_timeout(500)
                page.evaluate(axe_js)
                found = page.evaluate(
                    "axe.run().then(r => r.violations.filter(v => ['serious','critical'].includes(v.impact))"
                    ".map(v => v.id + ': ' + v.nodes.slice(0, 2).map(n => n.target.join(' ')).join(' | ')))"
                )
                problems += [f"axe {frag}: {v}" for v in found]
        problems += [f"page checks: {e}" for e in errors]
        browser.close()
    return problems


if __name__ == "__main__":
    found = check(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    server = fixture_server.serve()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    found += page_checks(f"http://127.0.0.1:{server.server_address[1]}")
    print(
        "\n".join(found)
        if found
        else "ui pages: ok (8 pages x 4 widths x dark/light, xss, session, cancel, redaction, axe)"
    )
    sys.exit(1 if found else 0)
