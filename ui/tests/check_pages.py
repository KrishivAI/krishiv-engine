# SPDX-License-Identifier: Apache-2.0
"""Snapshot check of the ten UI pages at phone, tablet and desktop widths, dark and light, of the pages the
embedded and the profiles console show, and of TPC-H Q9's profile in the console and in the offline viewer.

    uv run --with playwright python ui/tests/check_pages.py [SHOT_DIR]

Serves the built UI from the fixture server and fails when a page: scrolls sideways, loads anything from another
origin, logs a console error, lacks the self-hosted Red Hat fonts, loses its key text, or has a navigation target
under 40 px on a phone (the responsive checks ask for 44 px on touch screens). With SHOT_DIR it also writes a PNG per page and view (the visual record; the assertions
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
from urllib.parse import parse_qs, quote, urlparse

from playwright.sync_api import Page, expect, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

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
            "Peak memory",
            "not reported",
            "do not add up",
            "Advisor hints",
            "SPILL",
            "HashJoin spilled 128 MiB; give the query more memory.",
        ],
    ),
    ("history", "#/history", ["History", "DIVIDE_BY_ZERO", "Compare the two selected"]),
    (
        "sessions",
        "#/sessions",
        ["Sessions", "End session", "3 open", "ADBC Flight SQL", "for bi-gateway"],
    ),
    (
        "workload",
        "#/workload",
        ["Workload", "interactive", "nightly", "by priority · 2", "4 / 4", "removed"],
    ),
    (
        "advisor",
        "#/advisor",
        ["Advisor", "Partition by", "event_date", "100 of 120", "83.1%"],
    ),
    (
        "resources",
        "#/resources",
        [
            "Memory",
            "Caches",
            "root/query-1",
            "Python UDF workers",
            "5 of 64 worker processes, 3 busy",
            "5d0c7a52-etl",
            "acme",
        ],
    ),
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
        # A hash that is not `#/...` is an anchor inside the page, not a route: the page stays the path's.
        page.goto(f"{origin}/queries/{QID}?token=fixture#operators")
        try:
            page.get_by_text("HashJoin").first.wait_for(timeout=5000)
        except PlaywrightTimeout:
            problems.append(
                "deep link /queries/{id}#operators: an anchor was read as a route"
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

        # 2b. A link that names its query's session: the view sends the session with its read (the server signed
        # the link with it), and the same link without the session is not valid.
        for src, valid in (
            (f"{link}-session&session={quote(fixture_server.EMBED_SESSION)}", True),
            (f"{link}-session", False),
            # An empty qualifier is not "no qualifier": the view sends it and the server refuses it.
            (f"{link}&session=", False),
        ):
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            ctx.route("http://parent.test/**", serve_html(parent_html(src)))
            page = ctx.new_page()
            page.goto("http://parent.test/")
            want = "HashJoin" if valid else "This link is not valid"
            try:
                page.frame_locator("#f").get_by_text(want).first.wait_for(timeout=10000)
            except PlaywrightTimeout:
                problems.append(f"embed link {src!r}: {want!r} is not shown")
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

        # 4b. The allowed origins cannot be read: a theme sent as the frame loads is refused and logged, not held.
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
        ctx.route("**/api/internal/embed/config", lambda r: r.fulfill(status=500))
        page = ctx.new_page()
        _console(page, log)
        page.goto("http://parent.test/")
        page.frame_locator("#f").get_by_text("HashJoin").first.wait_for(timeout=10000)
        page.wait_for_timeout(500)
        if accent_of(page) == "#00ff00":
            problems.append(
                "a theme was applied although the allowed origins were not read"
            )
        if not any("not in krishiv.ui.allowedOrigins" in m for m in log):
            problems.append(
                f"a theme message held while the origins read failed was not logged: {log}"
            )
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
        page.get_by_role("button", name="orders", exact=True).click()
        page.get_by_text("o_orderkey").wait_for(timeout=10000)
        no_markup("catalog columns")
        # The editor's draft names a table by its three parts and a temporary view by its one.
        for name, want in (
            ("orders", "SELECT * FROM `spark_catalog`.`default`.`orders` LIMIT 100"),
            ("recent_orders", "SELECT * FROM `recent_orders` LIMIT 100"),
        ):
            page.evaluate("window.location.hash = '#/catalog'")
            for pick in ("spark_catalog", "default", name):
                page.get_by_role("button", name=pick, exact=True).click()
            page.get_by_role("link", name="Open in the SQL editor").click()
            page.locator(".cm-content").wait_for(timeout=10000)
            draft = page.locator(".cm-content").inner_text().strip()
            if draft != want:
                problems.append(f"catalog: the draft for {name} is {draft!r}")
        page.evaluate("window.location.hash = '#/sql'")
        editor = page.locator(".cm-content")
        editor.wait_for(timeout=10000)
        # Typing over a selection must not make the browser write a style attribute, which the policy blocks.
        blocked: list[str] = []
        page.on(
            "console",
            lambda m: blocked.append(m.text) if m.type == "error" else None,
        )
        editor.click()
        page.keyboard.press("Control+A")
        page.keyboard.type("SELECT 'typed over'")
        page.wait_for_timeout(300)
        typed = editor.inner_text().strip()
        if typed != "SELECT 'typed over'":
            problems.append(f"sql: typing over the selection left {typed!r}")
        problems += [f"sql: typing over a selection logs: {e}" for e in blocked]
        # Ctrl + Enter runs the statement, as the page says; it must not add a line to it instead.
        editor.click()
        page.keyboard.press("Control+A")
        page.keyboard.type("SELECT 1")
        page.keyboard.press("Control+Enter")
        try:
            page.get_by_text("finished", exact=True).first.wait_for(timeout=5000)
        except PlaywrightTimeout:
            problems.append("sql: Ctrl + Enter did not run the statement")
        kept = editor.inner_text().strip()
        if kept != "SELECT 1":
            problems.append(f"sql: Ctrl + Enter changed the statement to {kept!r}")
        page.reload()
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
        # A setting is found by its Spark name too, and a filter that matches nothing says so.
        box = page.get_by_placeholder("spark.sql")
        box.fill("spark.sql.shuffle")
        if not page.get_by_text(
            "krishiv.sql.shuffle.partitionsHint", exact=True
        ).first.is_visible():
            problems.append("environment: the Spark name does not find its setting")
        box.fill("no-such-setting")
        if not page.get_by_text("No setting matches.").first.is_visible():
            problems.append("environment: a filter with no match does not say so")
        box.fill("")
        # Pools of one name are one row on Resources.
        page.evaluate("window.location.hash = '#/resources'")
        page.locator("[data-pools]").first.wait_for(timeout=10000)
        named = page.locator(
            "tr", has=page.get_by_text("process/relation_cache", exact=True)
        ).count()
        pools = page.locator("[data-pools]").first.get_attribute("data-pools")
        row = page.locator("tr", has=page.locator("[data-pools]")).first.inner_text()
        if named != 1 or pools != "3" or "3.0 MiB" not in row or "8.0 MiB" not in row:
            problems.append(
                f"resources: relation_cache is {named} rows of {pools} pools: {row!r}"
            )
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


def shows(
    page: Page, texts: list[str], where: str = "body", timeout: int = 5000
) -> bool:
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
        (
            "#/resources",
            ["Memory pools and caches of this process"],
            ["Python UDF workers"],
        ),
        ("#/sql", ["Statements run in your session", "Run"], []),
        ("#/environment", ["The settings of this session", "***"], []),
        # The sessions of every client, the resource groups and the advice folded from every principal's
        # queries are a server's: their addresses open the first page.
        ("#/sessions", ["This process and the session"], ["End session"]),
        ("#/workload", ["This process and the session"], ["Scheduling"]),
        ("#/advisor", ["This process and the session"], ["Partition by"]),
    ],
    # Every address but a query's own page opens the history: the mode has no other page.
    "profiles": [
        ("", ["Saved profiles", "History", "DIVIDE_BY_ZERO", "Principal"], ["Health"]),
        ("#/health", ["DIVIDE_BY_ZERO"], ["Uptime", "Caches"]),
        ("#/queries", ["DIVIDE_BY_ZERO"], ["Refreshes once a second"]),
        ("#/sql", ["DIVIDE_BY_ZERO"], ["Ctrl or Cmd"]),
        ("#/environment", ["DIVIDE_BY_ZERO"], ["Secrets show"]),
        ("#/sessions", ["DIVIDE_BY_ZERO"], ["End session"]),
        ("#/workload", ["DIVIDE_BY_ZERO"], ["Scheduling"]),
        ("#/advisor", ["DIVIDE_BY_ZERO"], ["Partition by"]),
        (
            "#/queries/01932a40-1a00-7a01-8001-aaaaaaaaaa02",
            ["HashJoin", "← History"],
            ["← Queries"],
        ),
    ],
}
# The requests a console over stored profiles never makes: the pages behind them are not in the mode.
PROFILES_NEVER = (
    "/events",
    "/environment",
    "/catalog",
    "/sql",
    "/resources",
    "/sessions",
    "/groups",
    "/advice",
    "/udf-workers",
)


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
                    # Every page but a query's own, and but the three a server alone has.
                    if mode == "embedded" and len(links) != len(PAGES) - 4:
                        problems.append(f"embedded: the menu lists {links}")
                    tail = fixture_server.SESSION_ID[-8:]
                    if mode == "embedded" and tail not in page.inner_text("header"):
                        problems.append(
                            "embedded: the header does not show the session's id"
                        )
                problems += [f"{mode} {label}: page error: {e}" for e in errors]
                ctx.close()
            if mode == "embedded":
                problems += [
                    f"embedded: the page asked for {path}"
                    for path in sorted(set(server.asked))
                    if path.endswith(
                        ("/sessions", "/groups", "/advice", "/udf-workers")
                    )
                ]
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
        if not shows(
            page, ["The console did not load", "the console is starting"], "main"
        ):
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
    of one session has neither. Both filter by a part of the statement."""
    problems: list[str] = []
    # One link per listed query; the layout for the other width is in the page too, hidden.
    rows = "main a[href^='#/queries/']:visible"
    # Zero row links also describes "Loading…" and a failed read: no rows counts only once the list has loaded empty.
    loaded_empty = "main :text-is('No finished queries match.'):visible"
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
            statement = page.get_by_label("Statement", exact=True)
            if statement.count() != 1:
                problems.append(f"{mode}: History has no Statement filter")
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
                        if want == 0:
                            expect(page.locator(loaded_empty).first).to_be_visible(
                                timeout=5000
                            )
                        expect(page.locator(rows)).to_have_count(want, timeout=5000)
                    except AssertionError:
                        got = page.locator(rows).count()
                        problems.append(
                            f"history: {typed!r} lists {got} queries, not {want}"
                        )
                    field.fill("")
                    expect(page.locator(rows)).to_have_count(4, timeout=5000)
            # Any part of a statement, in either case and whatever the space around it, in every console.
            for typed, want in (
                ("LINEITEM", 2),
                (" from events ", 1),
                ("no such statement", 0),
            ):
                statement.fill(typed)
                try:
                    if want == 0:
                        expect(page.locator(loaded_empty).first).to_be_visible(
                            timeout=5000
                        )
                    expect(page.locator(rows)).to_have_count(want, timeout=5000)
                except AssertionError:
                    got = page.locator(rows).count()
                    problems.append(
                        f"{mode} history: statement {typed!r} lists {got} queries, not {want}"
                    )
                statement.fill("")
                expect(page.locator(rows)).to_have_count(4, timeout=5000)
            problems += [f"{mode} filters: page error: {e}" for e in errors]
            page.close()
            server.shutdown()
        browser.close()
    return problems


def compare_checks() -> list[str]:
    """History's comparison: two selected runs are compared, each named by its id and its session (two sessions
    may have run a query under one id); a run stored without a session is named by its id alone; a number one
    of the runs lacks has no change; a third selection replaces the older one and clears the table."""
    problems: list[str] = []
    runs = fixture_server.DATA["history"]
    session = runs[0]["session_id"]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        server = fixture_server.serve(mode="server")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/?token=fixture")
        page.locator("header nav a").first.wait_for(timeout=10000)
        page.evaluate("window.location.hash = '#/history'")

        def pick(run: dict) -> None:
            # The layout for the other width is in the page too, hidden.
            box = page.locator(
                f"main input[type=checkbox][aria-label*='({run['query_id'][-8:]})']:visible"
            )
            box.first.wait_for(timeout=10000)
            box.first.check()

        button = page.get_by_role("button", name="Compare the two selected")
        table = page.locator("main h2", has_text="Comparison")
        pick(runs[0])
        if not button.is_disabled():
            problems.append("compare: the button is enabled with one run selected")
        pick(runs[1])
        button.click()
        # (label, A, B, B - A) of the two runs of one session; the fixture's numbers.
        want = ("Wall time", "1.8 s", "2.6 s", "+770 ms")
        try:
            table.wait_for(timeout=5000)
            row = page.locator("main tr", has_text=want[0]).first.inner_text()
            if [c.strip() for c in row.split("\t")] != list(want):
                problems.append(f"compare: the wall time row is {row!r}")
        except PlaywrightTimeout:
            problems.append("compare: no comparison of two selected runs is shown")
        asked = parse_qs(server.compares[-1]) if server.compares else {}
        if asked != {
            "a": [runs[0]["query_id"]],
            "b": [runs[1]["query_id"]],
            "a_session": [session],
            "b_session": [session],
        }:
            problems.append(f"compare: two runs of a session were asked as {asked}")
        # A third selection drops the oldest; the stored run without a session is named by its id alone, and
        # it has no CPU time, so that row has no change.
        pick(runs[2])
        if table.count() != 0:
            problems.append(
                "compare: the old comparison stays after the selection changed"
            )
        button.click()
        try:
            table.wait_for(timeout=5000)
            row = page.locator("main tr", has_text="CPU time").first.inner_text()
            if [c.strip() for c in row.split("\t")][2:] != ["–", "–"]:
                problems.append(f"compare: the CPU time row is {row!r}")
            if (
                "−2.6 s"
                not in page.locator("main tr", has_text="Wall time").first.inner_text()
            ):
                problems.append("compare: a faster run B shows no negative change")
        except PlaywrightTimeout:
            problems.append("compare: no comparison with a run that has no session")
        asked = parse_qs(server.compares[-1]) if server.compares else {}
        if asked != {
            "a": [runs[1]["query_id"]],
            "b": [runs[2]["query_id"]],
            "a_session": [session],
        }:
            problems.append(f"compare: a run without a session was asked as {asked}")
        problems += [f"compare: page error: {e}" for e in errors]
        page.close()
        server.shutdown()
        browser.close()
    return problems


def editor_colour_checks() -> list[str]:
    """The SQL editor's syntax colours: a keyword, a string, a number and a comment each have their own colour,
    and every piece of the statement has a contrast of at least 4.5 against the editor's background, in dark and
    in light (CodeMirror's default colours are chosen for a white page)."""
    problems: list[str] = []
    statement = "SELECT 'a', 12 FROM t -- note"
    read = """el => {
        const rgb = c => (c.match(/[\\d.]+/g) || []).slice(0, 3).map(Number);
        const lum = c => {
            const [r, g, b] = rgb(c).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
            return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        };
        const back = getComputedStyle(el.closest('.cm-editor')).backgroundColor;
        const ratio = c => { const a = lum(c), b = lum(back); return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05); };
        const base = getComputedStyle(el).color;
        const parts = [...el.querySelectorAll('.cm-line span')].map(s => {
            const c = getComputedStyle(s).color;
            return { text: s.textContent, colour: c, ratio: ratio(c) };
        });
        return { back, base, baseRatio: ratio(base), parts };
    }"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        server = fixture_server.serve(mode="server")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        for scheme in ("dark", "light"):
            page = browser.new_page(
                viewport={"width": 1440, "height": 900}, color_scheme=scheme
            )
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/?token=fixture")
            page.locator("header nav a").first.wait_for(timeout=10000)
            page.evaluate("window.location.hash = '#/sql'")
            editor = page.locator(".cm-content")
            editor.wait_for(timeout=10000)
            editor.click()
            page.keyboard.press("Control+A")
            page.keyboard.type(statement)
            found = editor.evaluate(read)
            if found["baseRatio"] < 4.5:
                problems.append(
                    f"editor {scheme}: plain text {found['base']} on {found['back']} has contrast {found['baseRatio']:.2f}"
                )
            colours = {}
            for part in found["parts"]:
                colours[part["text"].strip()] = part["colour"]
                if part["ratio"] < 4.5:
                    problems.append(
                        f"editor {scheme}: {part['text']!r} {part['colour']} on {found['back']} has contrast {part['ratio']:.2f}"
                    )
            wanted = ["SELECT", "'a'", "12", "-- note"]
            missing = [w for w in wanted if w not in colours]
            if missing:
                problems.append(f"editor {scheme}: no coloured piece for {missing}")
            else:
                seen = [colours[w] for w in wanted]
                if len(set(seen)) != len(seen) or found["base"] in seen:
                    problems.append(
                        f"editor {scheme}: keyword, string, number and comment colours are {seen}, plain text {found['base']}"
                    )
                if colours["SELECT"] != colours.get("FROM"):
                    problems.append(
                        f"editor {scheme}: SELECT and FROM differ in colour"
                    )
            page.close()
        server.shutdown()
        browser.close()
    return problems


def workload_checks() -> list[str]:
    """A server console's Sessions and Workload pages and the filter of the live list: the ended sessions are
    off the list until asked for, the filter narrows it, ending a session takes two presses and says what was
    cancelled, a session's query count opens History on that session; the groups are indented under their
    parents and a server with none says so."""
    problems: list[str] = []
    server = fixture_server.serve()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # One row per session or group: the table's rows at this width (the cards for a narrower one are hidden),
    # without the one row an empty table shows its text in.
    rows = "main table tbody tr:visible:not(:has(td[colspan]))"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on(
            "dialog", lambda d: (errors.append(f"dialog: {d.message}"), d.dismiss())
        )
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/?token=fixture")
        page.locator("header nav a").first.wait_for(timeout=10000)

        def count(what: str, want: int) -> None:
            try:
                expect(page.locator(rows)).to_have_count(want, timeout=5000)
            except AssertionError:
                got = page.locator(rows).count()
                problems.append(f"{what}: {got} rows, not {want}")

        page.evaluate("window.location.hash = '#/sessions'")
        count("sessions: the open sessions", 3)
        if page.get_by_role("button", name="End session").count() != 2 or not shows(
            page, ["The console's editor"], "main"
        ):
            problems.append(
                "sessions: the editor's session is not marked, or offers End session"
            )
        if "idle_timeout" in page.inner_text("main"):
            problems.append(
                "sessions: an ended session is listed before it is asked for"
            )
        filter_box = page.get_by_label("Filter", exact=True)
        for typed, want in (("flight", 1), ("TEAM=RISK", 1), ("no-such-client", 0)):
            filter_box.fill(typed)
            count(f"sessions: the filter {typed!r}", want)
        filter_box.fill("")
        page.get_by_label("Show ended sessions").check()
        count("sessions: with the ended ones", 4)
        if not shows(page, ["expired", "idle_timeout"], "main"):
            problems.append("sessions: an ended session does not say how it ended")
        page.get_by_label("Show ended sessions").uncheck()
        count("sessions: the open sessions again", 3)

        # Ending a session: the first press asks, "Keep" takes it back, the second press ends it.
        bob = page.locator(rows).filter(has_text="bob")
        bob.get_by_role("button", name="End session").click()
        if server.closed:
            problems.append("sessions: one press ended a session")
        bob.get_by_role("button", name="Keep").click()
        if bob.get_by_role("button", name="End session").count() != 1:
            problems.append("sessions: Keep does not take the question back")
        bob.get_by_role("button", name="End session").click()
        bob.get_by_role("button", name="End it and cancel 1 running").click()
        if not shows(page, ["Ended the session of bob and cancelled 1 query."], "main"):
            problems.append("sessions: the page does not say what ending a session did")
        count("sessions: after one was ended", 2)
        if server.closed != [fixture_server.SESSION_B]:
            problems.append(f"sessions: the page ended {server.closed}")

        # A session's query count opens History on that session's queries.
        ann = page.locator(rows).filter(has_text="ann")
        ann.get_by_role(
            "button", name="The 14 queries of this session in History"
        ).click()
        page.wait_for_function("location.hash === '#/history'", timeout=5000)
        history_rows = "main a[href^='#/queries/']:visible"
        try:
            expect(page.locator(history_rows)).to_have_count(2, timeout=5000)
        except AssertionError:
            got = page.locator(history_rows).count()
            problems.append(f"history of a session: {got} queries, not 2")
        session_box = page.get_by_label("Session", exact=True)
        if session_box.input_value() != fixture_server.SESSION_A:
            problems.append("history of a session: the Session filter does not show it")
        session_box.fill("")
        expect(page.locator(history_rows)).to_have_count(4, timeout=5000)
        # The id was read once: coming back to History later lists everything.
        page.evaluate("window.location.hash = '#/health'")
        shows(page, ["uptime"], "main")
        page.evaluate("window.location.hash = '#/history'")
        expect(page.locator(history_rows)).to_have_count(4, timeout=5000)

        # The live list: the filter finds a query by its principal, its text or its session.
        page.evaluate("window.location.hash = '#/queries'")
        live = len(fixture_server.DATA["live"])
        count("queries: the live list", live)
        filter_box = page.get_by_label("Filter", exact=True)
        for typed, want in (
            ("bob", 1),
            ("L_RETURNFLAG", 1),
            (fixture_server.SESSION_A, 2),
            ("no-such-query", 0),
        ):
            filter_box.fill(typed)
            count(f"queries: the filter {typed!r}", want)
            shown = page.locator("[data-query-count]")
            if shown.inner_text() != f"{want} of {live} shown":
                problems.append(
                    f"queries: {typed!r} says {shown.inner_text()!r}, not '{want} of {live} shown'"
                )
        if not shows(page, ["No running query matches."], "main"):
            problems.append("queries: an empty filtered list is not said to be one")
        filter_box.fill("")
        count("queries: the live list again", live)

        # The groups: a parent before its children, each indented under it, a limit reached in bold.
        page.evaluate("window.location.hash = '#/workload'")
        count("workload: the groups", len(fixture_server.GROUPS))
        depths = page.locator("main table [data-depth]").evaluate_all(
            "es => es.map(e => [e.textContent.trim(), Number(e.dataset.depth), e.title])"
        )
        want_depths = [
            ["root", 0, "root"],
            ["interactive", 1, "root.interactive"],
            ["etl", 1, "root.etl"],
            ["nightly", 2, "root.etl.nightly"],
            ["adhoc", 1, "root.adhoc"],
        ]
        if depths != want_depths:
            problems.append(f"workload: the groups are {depths}")
        bold = page.locator("main table td span.font-bold").all_inner_texts()
        # `root.etl` runs as many as it may; `root.etl.nightly` also holds as many waiting as it may. A group
        # that may hold none waiting (`root.adhoc`, 0 / 0) is not "at its limit".
        if bold != ["4 / 4", "2 / 2", "5 / 5"]:
            problems.append(f"workload: the limits shown as reached are {bold}")
        problems += [f"workload: page error: {e}" for e in errors]
        if page.evaluate("window.__xss"):
            problems.append("workload: markup in a session's row ran")
        browser.close()
    server.shutdown()

    # A server started with no resource-group file: the page says so and shows no table.
    server = fixture_server.serve(groups=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(
            f"http://127.0.0.1:{server.server_address[1]}/?token=fixture#/workload"
        )
        try:
            page.locator("[data-no-groups]").wait_for(timeout=10000)
        except PlaywrightTimeout:
            problems.append("workload: a server with no groups does not say so")
        if page.locator("main table, main ul li").count():
            problems.append("workload: a server with no groups shows a list")
        browser.close()
    server.shutdown()
    return problems


def advisor_checks() -> list[str]:
    """A server console's Advisor page and the session a query's link carries: one row a table with its best
    column and the other candidates, a filter over the tables, the history read when the page opens and when
    asked (never on a timer), a server with no advice saying so; and a listed query's page and its Cancel name
    the row's session."""
    problems: list[str] = []
    server = fixture_server.serve()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rows = "main table tbody tr:visible:not(:has(td[colspan]))"
    advice_path = "/api/internal/advice"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/?token=fixture")
        page.locator("header nav a").first.wait_for(timeout=10000)

        def count(what: str, want: int) -> None:
            try:
                expect(page.locator(rows)).to_have_count(want, timeout=5000)
            except AssertionError:
                got = page.locator(rows).count()
                problems.append(f"{what}: {got} rows, not {want}")

        page.evaluate("window.location.hash = '#/advisor'")
        tables = fixture_server.ADVICE
        count("advisor: one row a table", len(tables))
        first = page.locator(rows).first.inner_text()
        for shown in (
            "spark_catalog.default.events",
            "event_date",
            "100 of 120",
            "365",
        ):
            if shown not in first:
                problems.append(f"advisor: the first table's row lacks {shown!r}")
        others = page.locator("main table [data-other-candidates] li").all_inner_texts()
        if (
            len(others) != 2
            or "region · 20 queries · 3 partitions · 11.1%" not in others[0]
        ):
            problems.append(f"advisor: the other candidates are {others}")
        # A share too small for one decimal still reads as a number, not as markup or an exponent.
        if len(others) == 2 and "0.1%" not in others[1]:
            problems.append(f"advisor: a small share reads {others[1]!r}")
        filter_box = page.get_by_label("Filter", exact=True)
        for typed, want in (("ORDERS", 1), ("spark_catalog", 2), ("no-such-table", 0)):
            filter_box.fill(typed)
            count(f"advisor: the filter {typed!r}", want)
            said = page.locator("[data-advice-count]").inner_text()
            if said != f"{want} of {len(tables)} tables shown":
                problems.append(f"advisor: {typed!r} says {said!r}")
        if not shows(page, ["No table matches."], "main"):
            problems.append("advisor: an empty filtered list is not said to be one")
        filter_box.fill("")
        count("advisor: the tables again", len(tables))
        # Reading the advice reads the history store: once for the page, and once more only when asked.
        page.wait_for_timeout(2500)
        if server.asked.count(advice_path) != 1:
            problems.append(
                f"advisor: the page read the advice {server.asked.count(advice_path)} times unasked"
            )
        page.get_by_role("button", name="Read the history again").click()
        try:
            page.wait_for_function(
                "() => !document.querySelector('main button[disabled]')", timeout=5000
            )
        except PlaywrightTimeout:
            problems.append("advisor: the page never finished reading again")
        if server.asked.count(advice_path) != 2:
            problems.append("advisor: asking again did not read the advice once more")
        if page.evaluate("window.__xss"):
            problems.append("advisor: markup in a column's name ran")

        # A listed query's page and its Cancel name the row's session: two sessions may share a query id.
        page.evaluate("window.location.hash = '#/queries'")
        bob = page.locator(rows).filter(has_text="bob")
        bob.get_by_role("button", name="Cancel").click()
        want = f"/cancel?session={fixture_server.SESSION_B}"
        if not shows(page, ["Cancelled "], "main"):
            problems.append("queries: Cancel never answered")
        if len(server.cancels) != 1 or not server.cancels[0].endswith(want):
            problems.append(f"queries: Cancel asked for {server.cancels}")
        bob.locator("a[href^='#/queries/']").click()
        if not shows(page, ["Operators"], "main"):
            problems.append("queries: a row's link does not open the query's page")
        want = f"?session={fixture_server.SESSION_B}"
        if not server.query_reads or not all(
            r.endswith(want) for r in server.query_reads
        ):
            problems.append(f"queries: the query's page read {server.query_reads}")
        # A row with no session (a profile stored before a restart) links the id alone, as before.
        try:
            with page.expect_response(
                lambda r: r.url.endswith(f"/api/internal/queries/{QID}"), timeout=5000
            ) as unqualified:
                page.evaluate(f"window.location.hash = '#/queries/{QID}'")
            read = unqualified.value.ok
        except PlaywrightTimeout:
            read = False
        if (
            not read
            or not shows(page, ["Operators"], "main")
            or "?" in server.query_reads[-1]
        ):
            problems.append(
                f"queries: a link without a session read {server.query_reads}"
            )
        problems += [f"advisor: page error: {e}" for e in errors]
        browser.close()
    server.shutdown()

    # A server whose history has no advice: the page says where advice comes from and shows no table.
    server = fixture_server.serve(advice=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(
            f"http://127.0.0.1:{server.server_address[1]}/?token=fixture#/advisor"
        )
        try:
            page.locator("[data-no-advice]").wait_for(timeout=10000)
        except PlaywrightTimeout:
            problems.append("advisor: a server with no advice does not say so")
        if page.locator("main table, main ul li, main input").count():
            problems.append("advisor: a server with no advice shows a list or a filter")
        browser.close()
    server.shutdown()
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


# The heatmap's lines: depth, name, value text, state, border style and background.
HEAT_ROWS = "main [data-testid='plan-tree'] > li"
HEAT = """
() => [...document.querySelectorAll("main [data-testid='plan-tree'] > li")].map(li => [
  Number(li.dataset.depth),
  li.querySelector("[data-part='name']").textContent,
  li.querySelector("[data-part='value']").textContent.replace(/\\s+/g, ' ').trim(),
  li.dataset.state,
  getComputedStyle(li).borderTopStyle,
  getComputedStyle(li).backgroundColor,
])
"""
CLEAR = ("rgba(0, 0, 0, 0)", "transparent")

# What the query page's fixture (`fixtures.json` `detail`) must show for each metric: depth, node, value.
# The plan: an aggregate with a sink and a source, an exchange no operator runs, a join with a build and a
# probe (and a child index that points nowhere), a filter, two scans, two headings with no id; the result sink
# has no node and comes last.
NODES = [
    (0, "HashAggregate"),
    (1, "Exchange"),
    (2, "HashJoin"),
    (3, "Filter"),
    (4, "FileScan"),
    (3, "Subquery"),
    (4, "SharedPlan"),
    (5, "FileScan"),
    (0, "Result"),
]
NR = "not reported"
HEATMAPS = {
    # Times add up over a node's operators; each has its share of the plan's total.
    "Compute time": ["600 ms 20.8%", NR, "1.7 s 58.8%", "90 ms 3.1%", "410 ms 14.2%", NR, NR, "90 ms 3.1%", "20.0 µs <0.1%"],
    # The source side's rows; a sink has none.
    "Rows out": ["5", NR, "15,000,000", "60,000,000", "75,000,000", NR, NR, "15,000,000", NR],
    # The largest peak of a node's operators; an operator that reserved nothing (the filter, the second scan,
    # the result) is not reported, never 0 B.
    "Peak memory": ["47.7 MiB", NR, "95.4 MiB", NR, "143.1 MiB", NR, NR, NR, NR],
    # The filter and the result waited 0 ns: a measured zero.
    "Waiting time": ["20 ms 1.5%", NR, "300 ms 22.4%", "0 ns 0.0%", "900 ms 67.2%", NR, NR, "120 ms 9.0%", "0 ns 0.0%"],
    # An operator that holds memory and spilled nothing is a measured zero; one that holds none cannot spill.
    "Spilled bytes": ["0 B 0.0%", NR, "128.0 MiB 100.0%", NR, "0 B 0.0%", NR, NR, NR, NR],
}  # fmt: skip
TIME_METRICS = ("Compute time", "Waiting time")


def tree_order(plan: list[dict]) -> list[dict]:
    """The nodes of a profile's plan, depth first from the root, as the page draws them."""
    out: list[dict] = []

    def walk(at: int) -> None:
        out.append(plan[at])
        for child in plan[at]["children"]:
            walk(child)

    walk(0)
    return out


def heatmap_checks(origin: str) -> list[str]:
    """The plan heatmap of the fixture's query, metric by metric: the tree's order and depths, each node's
    value (how its operators combine, the share of the total), and the three looks: a shaded value, a measured
    zero (dashed, no shade) and "not reported" (no shade, no figure). Then a query with no plan tree."""
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{origin}/?token=fixture#/queries/{fixture_server.Q9_ID[:-1]}1")
        page.locator(HEAT_ROWS).first.wait_for(timeout=10000)
        for label, values in HEATMAPS.items():
            page.get_by_role("button", name=label, exact=True).click()
            view = f"heatmap {label}"
            if (
                page.get_by_role("button", name=label, exact=True).get_attribute(
                    "aria-pressed"
                )
                != "true"
            ):
                problems.append(f"{view}: the metric's button is not pressed")
            rows = page.evaluate(HEAT)
            got = [(depth, name, value) for depth, name, value, *_ in rows]
            want = [(depth, name, value) for (depth, name), value in zip(NODES, values)]
            if got != want:
                problems.append(f"{view}: the heatmap is {got}, not {want}")
                continue
            for _, name, value, state, border, background in rows:
                want_state = (
                    "none"
                    if value == NR
                    else "zero"
                    if value.startswith("0 ")
                    else "value"
                )
                if state != want_state:
                    problems.append(
                        f"{view}: {name} ({value}) is drawn as {state}, not {want_state}"
                    )
                if (border == "dashed") != (want_state == "zero"):
                    problems.append(f"{view}: {name} ({value}) has a {border} border")
                # Not reported has no colour; a measured zero has the card's own, a value its shade.
                if (background in CLEAR) != (want_state == "none"):
                    problems.append(
                        f"{view}: {name} ({value}) has the background {background}"
                    )
            # One scale: the largest value has the strongest shade.
            alphas = page.evaluate(
                """() => [...document.querySelectorAll("main [data-testid='plan-tree'] > li[data-state='value']")].map(li => {
                  const c = document.createElement('canvas').getContext('2d');
                  c.fillStyle = getComputedStyle(li).backgroundColor; c.fillRect(0, 0, 1, 1);
                  return c.getImageData(0, 0, 1, 1).data[3];
                })"""
            )
            first = [r for r in rows if r[3] == "value"]
            largest = max(range(len(first)), key=lambda i: alphas[i])
            top = HEATMAP_TOP[label]
            if first[largest][1] != top:
                problems.append(
                    f"{view}: the strongest shade is on {first[largest][1]}, not {top}"
                )
            note = page.get_by_test_id("wall-note")
            if (note.count() == 1) != (label in TIME_METRICS):
                problems.append(
                    f"{view}: the wall-time sentence is shown {note.count()} times"
                )
        # The operator table follows the same rules: a source has no rows in, a sink no rows out, an operator
        # that reserved nothing no peak, and only a scan waits on I/O.
        table = page.locator("main tbody tr")
        cells = [
            [c.strip() for c in row.locator("td").all_inner_texts()]
            for row in table.all()
        ]
        want_cells = {
            "0 Result": ["5", "–", "20.0 µs", "– / 0 ns / 0 ns", "–", "–"],
            "5 Filter": [
                "75,000,000",
                "60,000,000",
                "90 ms",
                "– / 0 ns / 0 ns",
                "–",
                "–",
            ],
            "6 FileScan": [
                "–",
                "75,000,000",
                "410 ms",
                "900 ms / 0 ns / 0 ns",
                "143.1 MiB",
                "0 B",
            ],
            "7 FileScan": [
                "–",
                "15,000,000",
                "90 ms",
                "120 ms / 0 ns / 0 ns",
                "–",
                "–",
            ],
        }
        for row in cells:
            name = " ".join(row[0].split()[:2]) if row else ""
            if name in want_cells and row[1:] != want_cells[name]:
                problems.append(
                    f"operator table: {name} reads {row[1:]}, not {want_cells[name]}"
                )
            want_cells.pop(name, None)
        problems += [f"operator table: no row for {name}" for name in want_cells]
        if page.get_by_test_id("no-plan").count():
            problems.append("heatmap: a query with a plan says it has none")
        # An id two sessions ran a query under: no operators are given, and the page says why.
        page.goto(f"{origin}/?token=fixture#/queries/{fixture_server.SHARED_ID}")
        if not shows(page, ["Another session ran a query under this id"], "main"):
            problems.append("query detail: withheld operators are not explained")
        if shows(page, ["No operator metrics recorded"], "main", timeout=500):
            problems.append("query detail: withheld operators read as none recorded")
        # No plan tree: the operators in id order, at one depth, and a sentence that says why.
        page.goto(f"{origin}/?token=fixture#/queries/{fixture_server.NO_PLAN_ID}")
        page.get_by_test_id("no-plan").wait_for(timeout=10000)
        flat = [(depth, name, value) for depth, name, value, *_ in page.evaluate(HEAT)]
        want_flat = [
            (0, "Result", "20.0 µs <0.1%"),
            (0, "HashAggregate", "400 ms 13.8%"),
            (0, "HashAggregate", "200 ms 6.9%"),
            (0, "HashJoinBuild", "500 ms 17.3%"),
            (0, "HashJoinProbe", "1.2 s 41.5%"),
            (0, "Filter", "90 ms 3.1%"),
            (0, "FileScan", "410 ms 14.2%"),
            (0, "FileScan", "90 ms 3.1%"),
        ]
        if flat != want_flat:
            problems.append(f"heatmap without a plan: {flat}, not {want_flat}")
        # Nothing is being stored for it: the page says so at once and does not ask again.
        said = page.get_by_test_id("no-plan").inner_text()
        if "No plan tree is stored for this query" not in said:
            problems.append(f"heatmap without a plan: the page says {said!r}")
        label = page.get_by_test_id("plan-tree").get_attribute("aria-label")
        if label != "Operators shaded by compute time":
            problems.append(f"heatmap without a plan: the list is labelled {label!r}")
        asked = fixture_server.NO_PLAN_ASKED[0]
        page.wait_for_timeout(2500)
        if fixture_server.NO_PLAN_ASKED[0] != asked:
            problems.append(
                f"heatmap without a plan: the page asked {fixture_server.NO_PLAN_ASKED[0] - asked} more times"
            )
        # A query watched to its end: the first answers after it ended have no plan (the profile is written
        # a moment later), the page asks again, and the tree is drawn when the plan arrives.
        fixture_server.LATE_PLAN_ASKED[0] = 0
        page.goto(f"{origin}/?token=fixture#/queries/{fixture_server.LATE_PLAN_ID}")
        page.get_by_test_id("no-plan").wait_for(timeout=10000)
        said = page.get_by_test_id("no-plan").inner_text()
        if "being stored" not in said:
            problems.append(
                f"heatmap: while the plan is pending the page says {said!r}"
            )
        try:
            page.locator(f"{HEAT_ROWS}[data-depth='5']").wait_for(timeout=10000)
        except PlaywrightTimeout:
            problems.append(
                "heatmap: a plan that arrives after the query ended is never drawn"
            )
        else:
            late = [(depth, name) for depth, name, *_ in page.evaluate(HEAT)]
            if late != NODES:
                problems.append(
                    f"heatmap: the late plan is drawn as {late}, not {NODES}"
                )
            if page.get_by_test_id("no-plan").count():
                problems.append(
                    "heatmap: the page still says there is no plan after drawing it"
                )
        problems += [f"heatmap: page error: {e}" for e in errors]
        browser.close()
    return problems


# The node with the largest value of each metric.
HEATMAP_TOP = {
    "Compute time": "HashJoin",
    "Rows out": "FileScan",
    "Peak memory": "FileScan",
    "Waiting time": "FileScan",
    "Spilled bytes": "HashJoin",
}


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
            try:
                page.locator(HEAT_ROWS).first.wait_for(timeout=10000)
            except PlaywrightTimeout:
                problems.append(f"{view}: the page shows no heatmap")
            heat = page.evaluate(HEAT)
            # The heatmap is the profile's plan as a tree (the offline viewer's tree), then the operator no node
            # claims: the result sink.
            drawn = "".join(f"{'  ' * depth}{name}\n" for depth, name, *_ in heat)
            if drawn != tree + "Result\n":
                problems.append(
                    f"{view}: the heatmap is not python/tests/fixtures/q09-plan-tree.txt and the result: {drawn!r}"
                )
            # A node no operator runs (an exchange) is not reported; every other one has a compute time.
            ran = {o["node"] for o in operators}
            want_states = [
                "none" if n["id"] not in ran else "value"
                for n in tree_order(doc["plan"])
            ] + ["value"]
            if [state for *_, state, _, _ in heat] != want_states:
                problems.append(
                    f"{view}: the heatmap's states are {[h[3] for h in heat]}, not {want_states}"
                )
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
    found += compare_checks()
    found += editor_colour_checks()
    found += workload_checks()
    found += advisor_checks()
    found += heatmap_checks(f"http://127.0.0.1:{server.server_address[1]}")
    found += q9_checks(shots)
    found += responsive_checks(shots / "responsive" if shots is not None else None)
    print(
        "\n".join(found)
        if found
        else "ui pages: ok (11 pages x 4 widths x dark/light, xss, session, cancel, redaction, axe, deep links, embed, "
        "embedded and profiles consoles, operator, history filters and comparison, sessions and workload, the advisor and session links, the plan heatmap, "
        "TPC-H Q9's profile, "
        "responsive: 3 consoles x 13 screen sizes, touch targets, SQL editor on a phone, text size, motion)"
    )
    sys.exit(1 if found else 0)
