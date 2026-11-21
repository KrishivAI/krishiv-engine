# SPDX-License-Identifier: Apache-2.0
"""Snapshot check of the eight UI pages at phone, tablet and desktop widths, dark and light, of the pages the
embedded and the profiles console show, and of TPC-H Q9's profile in the console and in the offline viewer.

    uv run --with playwright python ui/tests/check_pages.py [SHOT_DIR]

Serves the built UI from the fixture server and fails when a page: scrolls sideways, loads anything from another
origin, logs a console error, lacks the self-hosted Red Hat fonts, loses its key text, or has a navigation target
under 40 px on a phone. With SHOT_DIR it also writes a PNG per page and view (the visual record; the assertions
are the check, since pixels differ between machines). It ends with the responsive and device checks of
`check_responsive.py` (13 screen sizes from a phone to an ultrawide monitor, touch targets, the SQL editor on a
phone, text size, motion), whose pictures go to SHOT_DIR/responsive.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
from html import escape
from pathlib import Path
from urllib.parse import quote, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
import fixture_server
from check_responsive import responsive_checks

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
        [
            "HashJoin",
            "128.0 MiB",
            "Operators",
            "Advisor hints",
            "SPILL",
            "HashJoin spilled 128 MiB; give the query more memory.",
        ],
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

# At desktop width a table shows all its columns: its box scrolls sideways only on a narrower screen.
CUT_TABLES = """
() => [...document.querySelectorAll('[data-table-box]')]
  .filter(e => e.offsetParent !== null && e.scrollWidth > e.clientWidth)
  .map(e => e.scrollWidth - e.clientWidth)
"""


QID = "01932a40-1a00-7a01-8001-aaaaaaaaaa02"
SAMPLE_THEME = {"mode": "dark", "tokens": {"accent": "#ff0000", "radius-md": "2px"}}


def _console(page: object, sink: list[str]) -> None:
    page.on("console", lambda m: sink.append(f"{m.type}: {m.text}"))  # type: ignore[attr-defined]
    page.on("pageerror", lambda e: sink.append(f"error: {e}"))  # type: ignore[attr-defined]


def serve_html(body: str):
    """A route handler answering every request with `body` (a closure: Playwright passes the handler two arguments)."""
    return lambda route: route.fulfill(status=200, content_type="text/html", body=body)


def embed_checks(origin: str, shots: Path | None) -> list[str]:
    """Deep links navigate; the embedded profile renders when framed from an allowed origin and not from another,
    at three widths with the default and a sample theme; theme messages are applied only when valid and from an
    allowed origin, also one sent as the frame loads; the embedded view has no header or navigation."""
    problems: list[str] = []
    link = f"{origin}/queries/{QID}/profile?embed=1&exp=9999999999&sig=ok"

    def parent_html(src: str, onload: str = "") -> str:
        return f'<!doctype html><iframe id="f" src="{src}" style="width:100%;height:700px;border:0" onload="{escape(onload)}"></iframe>'

    with sync_playwright() as p:
        # The fixture is on loopback and the embedding consoles are made-up public names: Chromium's local-network
        # checks would refuse that frame, which is not what this check is about.
        browser = p.chromium.launch(
            args=[
                "--disable-features=LocalNetworkAccessChecks,PrivateNetworkAccessSendPreflights,BlockInsecurePrivateNetworkRequests"
            ]
        )
        # 1. A deep link navigates away: the header and the back link of the detail page still work.
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto(f"{origin}/queries/{QID}?token=fixture")
        page.get_by_text("HashJoin").first.wait_for(timeout=10000)
        page.get_by_role("link", name="Health").first.click()
        try:
            page.get_by_text("Uptime").first.wait_for(timeout=5000)
        except PlaywrightTimeout:
            problems.append(
                "deep link /queries/{id}: the header's Health link does not leave the page"
            )
        page.get_by_role("link", name="Queries").first.click()
        try:
            page.get_by_text("SELECT l_returnflag").first.wait_for(timeout=5000)
        except PlaywrightTimeout:
            problems.append(
                "deep link: the header's Queries link does not show the query list"
            )
        page.goto(f"{origin}/sql?token=fixture")
        page.get_by_role("link", name="Environment").first.click()
        try:
            page.get_by_text("krishiv.auth.token").first.wait_for(timeout=5000)
        except PlaywrightTimeout:
            problems.append(
                "deep link /sql: the Environment link does not leave the page"
            )
        ctx.close()

        # 2. Framed from the allowed origin: renders, with no header or navigation, at three widths.
        for label, width, height in (
            ("phone", 390, 844),
            ("tablet", 820, 1180),
            ("desktop", 1440, 900),
        ):
            for theme_name, theme in (("default", None), ("sample", SAMPLE_THEME)):
                src = link
                if theme is not None:
                    src += "&theme=" + quote(json.dumps(theme))
                ctx = browser.new_context(viewport={"width": width, "height": height})
                log: list[str] = []
                body = parent_html(src)
                ctx.route("http://parent.test/**", serve_html(body))
                page = ctx.new_page()
                _console(page, log)
                page.goto("http://parent.test/")
                frame = page.frame_locator("#f")
                view = f"embed {label} {theme_name}"
                try:
                    frame.get_by_text("HashJoin").first.wait_for(timeout=10000)
                except PlaywrightTimeout:
                    problems.append(f"{view}: the framed profile does not render")
                    ctx.close()
                    continue
                if frame.locator("header, nav").count() != 0:
                    problems.append(
                        f"{view}: the embedded view shows a header or navigation"
                    )
                if frame.get_by_text("Powered by Krishiv").count() != 1:
                    problems.append(f"{view}: the attribution mark is not shown")
                accent = frame.locator("html").evaluate(
                    "e => getComputedStyle(e).getPropertyValue('--k-accent').trim()"
                )
                if (accent == "#ff0000") != (theme is not None):
                    problems.append(f"{view}: --k-accent is {accent!r}")
                if (
                    theme is not None
                    and frame.locator("html").get_attribute("data-theme") != "dark"
                ):
                    problems.append(f"{view}: the theme's mode was not applied")
                if shots is not None:
                    shots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(
                        path=str(shots / f"embed-{label}-{theme_name}.png"),
                        full_page=True,
                    )
                problems += [
                    f"{view}: {m}" for m in log if m.startswith(("error", "warning"))
                ]
                ctx.close()

        # 3. Framed from an origin that is not allowed: the browser blocks it (CSP frame-ancestors).
        ctx = browser.new_context()
        ctx.route(
            "http://blocked.test/**",
            lambda r: r.fulfill(
                status=200, content_type="text/html", body=parent_html(link)
            ),
        )
        page = ctx.new_page()
        page.goto("http://blocked.test/")
        page.wait_for_timeout(1500)
        framed = [f for f in page.frames if f != page.main_frame]
        if not framed or not framed[0].url.startswith("chrome-error:"):
            problems.append(
                f"embed from a disallowed origin was not blocked: {[f.url for f in framed]}"
            )
        ctx.close()

        # 4. Theme messages: valid from the allowed origin (sent as the frame loads) applies; the others do not.
        def message(theme: object) -> str:
            return f'this.contentWindow.postMessage({{type:"krishiv.theme",theme:{json.dumps(theme)}}},"*")'

        def accent_of(page: object) -> str:
            return (
                page.frame_locator("#f")
                .locator("html")
                .evaluate(  # type: ignore[attr-defined]
                    "e => getComputedStyle(e).getPropertyValue('--k-accent').trim()"
                )
            )

        ctx = browser.new_context()
        log = []
        ctx.route(
            "http://parent.test/**",
            lambda r: r.fulfill(
                status=200,
                content_type="text/html",
                body=parent_html(link, message({"tokens": {"accent": "#00ff00"}})),
            ),
        )
        page = ctx.new_page()
        _console(page, log)
        page.goto("http://parent.test/")
        page.frame_locator("#f").get_by_text("HashJoin").first.wait_for(timeout=10000)
        page.wait_for_timeout(500)
        if accent_of(page) != "#00ff00":
            problems.append(
                f"a theme sent as the frame loads was not applied (accent {accent_of(page)!r})"
            )
        for bad in (
            {"tokens": {"accent": "url(http://x/y)"}},
            {"tokens": {"nope": "#fff"}},
            {"mode": "sepia"},
        ):
            page.evaluate(
                f'(()=>{{const f=document.getElementById("f");{message(bad).replace("this.", "f.")}}})()'
            )
        page.wait_for_timeout(500)
        if accent_of(page) != "#00ff00":
            problems.append(
                f"an invalid theme changed the page (accent {accent_of(page)!r})"
            )
        if sum("ignored an invalid theme message" in m for m in log) != 3:
            problems.append(f"invalid theme messages were not logged: {log}")
        ctx.close()
        ctx = browser.new_context()
        log = []
        ctx.route(
            "http://other.test/**",
            lambda r: r.fulfill(
                status=200,
                content_type="text/html",
                body=parent_html(link, message({"tokens": {"accent": "#00ff00"}})),
            ),
        )
        page = ctx.new_page()
        _console(page, log)
        page.goto("http://other.test/")
        page.frame_locator("#f").get_by_text("HashJoin").first.wait_for(timeout=10000)
        page.wait_for_timeout(500)
        if accent_of(page) == "#00ff00":
            problems.append(
                "a theme message from an origin that is not allowed was applied"
            )
        if not any("not in krishiv.ui.allowedOrigins" in m for m in log):
            problems.append(f"a message from a disallowed origin was not logged: {log}")
        ctx.close()

        # 5. A theme in the link that is not valid is ignored and logged.
        ctx = browser.new_context()
        log = []
        bad_link = link + "&theme=" + quote(json.dumps({"tokens": {"accent": "red"}}))
        ctx.route(
            "http://parent.test/**",
            lambda r: r.fulfill(
                status=200, content_type="text/html", body=parent_html(bad_link)
            ),
        )
        page = ctx.new_page()
        _console(page, log)
        page.goto("http://parent.test/")
        page.frame_locator("#f").get_by_text("HashJoin").first.wait_for(timeout=10000)
        if not any("ignored the theme in the link" in m for m in log):
            problems.append(f"an invalid theme in the link was not logged: {log}")
        ctx.close()
        browser.close()
    return problems


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
                    if label == "desktop":
                        problems += [
                            f"{view}: a table is {w}px wider than its box, so its last columns are cut"
                            for w in page.evaluate(CUT_TABLES)
                        ]
                    body = page.inner_text("body")
                    if name == "query-detail":
                        hints = page.get_by_test_id("advisor-hints").all_inner_texts()
                        if (
                            len(hints) != 1
                            or "give the query more memory" not in hints[0]
                        ):
                            problems.append(
                                f"{view}: the advisor-hints section is {hints!r}"
                            )
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
        # The name carries the port: a browser keeps cookies per host name, so two consoles on 127.0.0.1 need two.
        name = f"krishiv_ui_token_{urlparse(origin).port}"
        cookies = [c for c in ctx.cookies() if c["name"].startswith("krishiv_ui_token")]
        if [c["name"] for c in cookies] != [name]:
            problems.append(f"expected one {name} cookie, got {cookies}")
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


def shows(page: Page, texts: list[str], where: str = "body", timeout: int = 5000) -> bool:
    """Waits until `where` shows every one of `texts` (any case); whether it did within `timeout` ms."""
    try:
        page.wait_for_function(
            "([sel, ts]) => { const b = document.querySelector(sel).innerText.toLowerCase();"
            " return ts.every(t => b.includes(t)); }",
            arg=[where, [t.lower() for t in texts]],
            timeout=timeout,
        )
    except PlaywrightTimeout:
        return False
    return True


# What each mode's console shows and leaves out, beyond the server console the checks above cover:
# (page, text that must be there, text that must not be).
MODE_PAGES = {
    "embedded": [
        (
            "#/health",
            ["This session", "This process and the session", "Uptime"],
            ["Workers"],
        ),
        ("#/queries", ["SELECT l_returnflag", "Cancel"], ["Principal"]),
        ("#/history", ["DIVIDE_BY_ZERO", "State"], ["Principal"]),
        ("#/resources", ["Memory pools and caches of this process"], []),
        ("#/sql", ["Statements run in your session", "Run"], []),
        ("#/environment", ["The settings of this session", "***"], []),
    ],
    # Every address but a query's own page opens the history: the mode has no other page.
    "profiles": [
        ("", ["Saved profiles", "History", "DIVIDE_BY_ZERO", "Principal"], ["Health"]),
        ("#/health", ["DIVIDE_BY_ZERO"], ["Uptime", "Caches"]),
        ("#/queries", ["DIVIDE_BY_ZERO"], ["Refreshes once a second"]),
        ("#/sql", ["DIVIDE_BY_ZERO"], ["Ctrl or Cmd"]),
        ("#/environment", ["DIVIDE_BY_ZERO"], ["Secrets show"]),
        (
            "#/queries/01932a40-1a00-7a01-8001-aaaaaaaaaa02",
            ["HashJoin", "← History"],
            ["← Queries"],
        ),
    ],
}
# The requests a console over stored profiles never makes: the pages behind them are not in the mode.
PROFILES_NEVER = ("/events", "/environment", "/catalog", "/sql", "/resources")


def mode_checks() -> list[str]:
    """The embedded and the profiles console: each shows its own pages and marks which console it is."""
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for mode, pages in MODE_PAGES.items():
            server = fixture_server.serve(mode=mode)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            origin = f"http://127.0.0.1:{server.server_address[1]}"
            for label, width, height in (VIEWPORTS[0], VIEWPORTS[-1]):
                ctx = browser.new_context(
                    viewport={"width": width, "height": height},
                    has_touch=label in PHONES,
                    is_mobile=label in PHONES,
                )
                page = ctx.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                page.goto(f"{origin}/?token=fixture")
                page.locator(f'[data-console-mode="{mode}"]').wait_for(timeout=10000)
                for frag, wanted, unwanted in pages:
                    view = f"{mode} {frag or '(no page)'} {label}"
                    page.evaluate(f"window.location.hash = {frag!r}")
                    # The page of this address, not the one before it or its loading state: the texts it
                    # must show are there. A page that never shows them is reported just below.
                    shows(page, wanted)
                    body = page.inner_text("main").lower()
                    if label == "desktop":
                        body += page.inner_text("header").lower()
                        problems += [
                            f"{view}: missing text {t!r}"
                            for t in wanted
                            if t.lower() not in body
                        ]
                    main = page.inner_text("main").lower()
                    problems += [
                        f"{view}: shows {t!r}, which this mode does not have"
                        for t in unwanted
                        if t.lower() in main
                    ]
                    wide = page.evaluate(
                        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
                    )
                    if wide > 0:
                        problems.append(
                            f"{view}: the page scrolls sideways ({wide}px too wide)"
                        )
                    if label in PHONES:
                        problems += [
                            f"{view}: touch target under 40 px: {s}"
                            for s in page.evaluate(SMALL_TARGETS)
                        ]
                if label == "desktop":
                    links = page.locator("header nav a").all_inner_texts()
                    if mode == "profiles" and links != ["History"]:
                        problems.append(
                            f"profiles: the menu lists {links}, not only History"
                        )
                    if mode == "embedded" and len(links) != len(PAGES) - 1:
                        problems.append(f"embedded: the menu lists {links}")
                    tail = fixture_server.SESSION_ID[-8:]
                    if mode == "embedded" and tail not in page.inner_text("header"):
                        problems.append(
                            "embedded: the header does not show the session's id"
                        )
                problems += [f"{mode} {label}: page error: {e}" for e in errors]
                ctx.close()
            if mode == "profiles":
                problems += [
                    f"profiles: the page asked for {path}"
                    for path in sorted(set(server.asked))
                    if path.endswith(PROFILES_NEVER)
                    or "/catalog/" in path
                    or path == "/api/internal/queries"
                ]
            server.shutdown()
        browser.close()
    return problems


def start_checks() -> list[str]:
    """A console whose first answer fails after the token was taken: it says so and offers a retry, and
    does not ask for the token again."""
    problems: list[str] = []
    server = fixture_server.serve(health_failures=1)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/?token=fixture")
        if not shows(page, ["The console did not load", "the console is starting"], "main"):
            problems.append("start: a failed first answer is not shown as one")
        main = page.inner_text("main")
        if "Open the address" in main:
            problems.append("start: a failed first answer asks for the token again")
        page.get_by_role("button", name="Try again").click()
        if not shows(page, ["uptime"], "main"):
            problems.append("start: the retry does not open the console's first page")
        browser.close()
    server.shutdown()
    return problems


def filter_checks() -> list[str]:
    """History's filters by client type and tag: a server console has them and they narrow the list; the console
    of one session has neither."""
    problems: list[str] = []
    # One link per listed query; the layout for the other width is in the page too, hidden.
    rows = "main a[href^='#/queries/']:visible"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for mode in ("server", "embedded"):
            server = fixture_server.serve(mode=mode)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/?token=fixture")
            # A server console carries no mode mark: its menu is the sign that the mode is read.
            page.locator("header nav a").first.wait_for(timeout=10000)
            page.evaluate("window.location.hash = '#/history'")
            page.locator(rows).first.wait_for(timeout=10000)
            fields = [page.get_by_label(n, exact=True) for n in ("Client", "Tag")]
            if mode == "embedded":
                if any(f.count() for f in fields):
                    problems.append("embedded: History has a Client or Tag filter")
            else:
                main = page.inner_text("main")
                for shown in ("krishiv-cli/0.2.1", "team=risk", "nightly"):
                    if shown not in main:
                        problems.append(f"history: a row does not show {shown!r}")
                for field, typed, want in (
                    (fields[0], "PYTHON", 1),
                    (fields[1], "team=risk", 2),
                    (fields[1], "team", 3),
                    (fields[1], "no-such-tag", 0),
                ):
                    field.fill(typed)
                    # Counted through a locator: the console's policy refuses a script given as text.
                    try:
                        expect(page.locator(rows)).to_have_count(want, timeout=5000)
                    except AssertionError:
                        got = page.locator(rows).count()
                        problems.append(
                            f"history: {typed!r} lists {got} queries, not {want}"
                        )
                    field.fill("")
                    expect(page.locator(rows)).to_have_count(4, timeout=5000)
            problems += [f"{mode} filters: page error: {e}" for e in errors]
            page.close()
            server.shutdown()
        browser.close()
    return problems


def role_checks() -> list[str]:
    """A server console opened by an operator: no Environment page, and its address opens the first page."""
    problems: list[str] = []
    server = fixture_server.serve(role="operator")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/?token=fixture")
        page.locator("header nav a").first.wait_for(timeout=10000)
        links = page.locator("header nav a").all_inner_texts()
        if "Environment" in links or len(links) != len(PAGES) - 2:
            problems.append(f"operator: the menu lists {links}")
        page.evaluate("window.location.hash = '#/environment'")
        shows(page, ["uptime"], "main")
        main = page.inner_text("main").lower()
        if "uptime" not in main or "secrets show" in main:
            problems.append("operator: #/environment does not open the Health page")
        problems += [f"operator: page error: {e}" for e in errors]
        browser.close()
    if "/api/internal/environment" in server.asked:
        problems.append("operator: the page asked for the settings")
    server.shutdown()
    return problems


# The plan tree of the offline viewer's page: each node's name under two spaces per level, as
# `python/tests/fixtures/q09-plan-tree.txt` writes it.
VIEWER_TREE = """
() => [...document.querySelectorAll('.kp span.name')].map(e => {
  let depth = -1;
  for (let a = e.parentElement; a; a = a.parentElement) if (a.tagName === 'UL') depth++;
  return '  '.repeat(depth) + e.textContent;
}).join('\\n') + '\\n'
"""


def q9_checks(shots: Path | None) -> list[str]:
    """TPC-H Q9's profile (`python/tests/fixtures/q09-profile.json`) in a browser: the console over stored profiles
    shows the query's page with every operator, and the offline viewer's page shows the plan as its tree, runs no
    script and asks the network for nothing."""
    problems: list[str] = []
    doc = fixture_server.Q9_PROFILE
    operators = sorted(doc["operators"], key=lambda o: o["id"])
    listed = [f"{o['id']} {o['name']}" for o in operators]
    hot = fixture_server.Q9_DETAIL["hot_operator"]
    server = fixture_server.serve(mode="profiles")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    repo = Path(__file__).parent.parent.parent
    # The viewer is one file that imports the standard library only, so it loads without the built package.
    spec = importlib.util.spec_from_file_location(
        "profile_view", repo / "python" / "krishiv" / "sql" / "profile_view.py"
    )
    if spec is None or spec.loader is None:
        return ["q9 viewer: python/krishiv/sql/profile_view.py does not load"]
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    tree = (repo / "python" / "tests" / "fixtures" / "q09-plan-tree.txt").read_text()
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch()
        saved = viewer.ProfileView(doc).save(Path(tmp) / "q09.html")
        for label, width, height in (VIEWPORTS[0], VIEWPORTS[-1]):
            ctx = browser.new_context(
                viewport={"width": width, "height": height},
                has_touch=label in PHONES,
                is_mobile=label in PHONES,
            )
            # The console's page of the query.
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.goto(f"{origin}/?token=fixture#/queries/{fixture_server.Q9_ID}")
            view = f"q9 console {label}"
            heat = page.locator(
                "main ol[aria-label^='Operators shaded'] li > span:first-child"
            )
            try:
                expect(heat).to_have_count(len(listed), timeout=10000)
            except AssertionError:
                problems.append(
                    f"{view}: the heatmap lists {heat.count()} operators, not {len(listed)}"
                )
            else:
                got = [" ".join(t.split()) for t in heat.all_inner_texts()]
                if got != listed:
                    problems.append(f"{view}: the heatmap lists {got}, not {listed}")
            rows = page.locator("main :is(tbody tr, ul > li):visible")
            if rows.count() != len(listed):
                problems.append(
                    f"{view}: the operator table has {rows.count()} rows, not {len(listed)}"
                )
            main = page.inner_text("main")
            for wanted in (
                "← History",
                "finished",
                "p_name LIKE '%green%'",
                "25 ms",
                f"the busiest one is highlighted: {hot['operator']}",
            ):
                if wanted not in main:
                    problems.append(f"{view}: missing text {wanted!r}")
            if "Warnings" in main:
                problems.append(f"{view}: a warning is shown for a run that has none")
            wide = page.evaluate(
                "document.documentElement.scrollWidth - document.documentElement.clientWidth"
            )
            if wide > 0:
                problems.append(
                    f"{view}: the page scrolls sideways ({wide}px too wide)"
                )
            if shots is not None:
                shots.mkdir(parents=True, exist_ok=True)
                page.screenshot(
                    path=str(shots / f"q09-console-{label}.png"), full_page=True
                )
            problems += [f"{view}: page error: {e}" for e in errors]
            page.close()
            # The offline viewer's page, opened from its file.
            page = ctx.new_page()
            asked: list[str] = []
            page.on("request", lambda r, asked=asked: asked.append(r.url))
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.goto(saved.as_uri())
            view = f"q9 viewer {label}"
            if page.evaluate(VIEWER_TREE) != tree:
                problems.append(
                    f"{view}: the plan tree is not python/tests/fixtures/q09-plan-tree.txt"
                )
            for selector, want in (
                (".kp .op", len(operators)),
                (".kp table tr", len(doc["pipelines"])),
                ("script", 0),
            ):
                got_count = page.locator(selector).count()
                if got_count != want:
                    problems.append(
                        f"{view}: {got_count} elements match {selector!r}, not {want}"
                    )
            if asked != [saved.as_uri()]:
                problems.append(f"{view}: the page asked for {asked}")
            if shots is not None:
                page.screenshot(
                    path=str(shots / f"q09-viewer-{label}.png"), full_page=True
                )
            ctx.close()
        browser.close()
    server.shutdown()
    return problems


if __name__ == "__main__":
    shots = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    found = check(shots)
    server = fixture_server.serve()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    found += page_checks(f"http://127.0.0.1:{server.server_address[1]}")
    found += embed_checks(
        f"http://127.0.0.1:{server.server_address[1]}",
        shots,
    )
    found += mode_checks()
    found += role_checks()
    found += start_checks()
    found += filter_checks()
    found += q9_checks(shots)
    found += responsive_checks(shots / "responsive" if shots is not None else None)
    print(
        "\n".join(found)
        if found
        else "ui pages: ok (8 pages x 4 widths x dark/light, xss, session, cancel, redaction, axe, deep links, embed, "
        "embedded and profiles consoles, operator, history filters, TPC-H Q9's profile, "
        "responsive: 3 consoles x 13 screen sizes, touch targets, SQL editor on a phone, text size, motion)"
    )
    sys.exit(1 if found else 0)
