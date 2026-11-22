# SPDX-License-Identifier: Apache-2.0
"""Responsive and device checks of the console: every page of the three consoles (server, embedded, saved
profiles), dark and light, from a 360 px phone to a 5120 px ultrawide monitor.

    uv run --with playwright python ui/tests/check_responsive.py [SHOT_DIR]

`check_pages.py` runs these too (so CI does). Per page and screen size it fails when: the page is wider than the
screen or anything is cut at the screen's edge; a table neither fits nor scrolls inside its own box (from 1280 px
it must fit); on a touch screen a button, link or field is under 44 x 44 px; the Cancel action of a running query
is not on the screen's layout; on an ultrawide monitor the content is not centred, the top bar does not line up
with it, text or a field runs wider than a readable line, or a table grows past 2,000 px. It also checks the
menu (opens, keeps the focus, Esc closes it), the SQL editor on a phone with the keyboard up, a large text size,
a minimum font size, reduced motion and the two colour schemes. With SHOT_DIR it writes a PNG per page for the
phone, tablet, desktop and ultrawide sizes.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

sys.path.insert(0, str(Path(__file__).parent))
import fixture_server

# (name, width, height). Touch screens come first; the rest are checked with a mouse.
TOUCH = [
    ("small-phone", 360, 740),
    ("phone", 390, 844),
    ("phone-landscape", 844, 390),
    ("tablet", 768, 1024),
    ("tablet-landscape", 1024, 768),
]
MOUSE = [
    # A notebook's side pane or a narrow window beside an editor.
    ("narrow-pane", 700, 500),
    ("laptop", 1280, 800),
    ("desktop", 1440, 900),
    # Half of a 3440 px monitor: a window snapped to one side.
    ("half-ultrawide", 1720, 1440),
    ("full-hd", 1920, 1080),
    ("ultrawide-1080", 2560, 1080),
    ("ultrawide", 3440, 1440),
    ("super-ultrawide", 5120, 1440),
]
# The sizes photographed with SHOT_DIR.
SHOT_SIZES = ("phone", "tablet", "desktop", "ultrawide")
# From this width a table shows all its columns; below it, it may scroll inside its box.
TABLES_FIT_FROM = 1280
ULTRAWIDE_FROM = 2560
DETAIL = "#/queries/01932a40-1a00-7a01-8001-aaaaaaaaaa02"
# (name, address, a text that shows the page has its data).
ALL_PAGES = [
    ("health", "#/health", "Uptime"),
    ("queries", "#/queries", "SELECT l_returnflag"),
    ("query-detail", DETAIL, "Operators"),
    ("history", "#/history", "DIVIDE_BY_ZERO"),
    ("resources", "#/resources", "root/query-1"),
    ("catalog", "#/catalog", "spark_catalog"),
    ("sql", "#/sql", "Run"),
    ("environment", "#/environment", "krishiv.auth.token"),
]
# The three pages a server console alone has.
SERVER_PAGES = [
    ("sessions", "#/sessions", "ADBC Flight SQL"),
    ("workload", "#/workload", "nightly"),
    ("advisor", "#/advisor", "event_date"),
]
MODE_PAGES = {
    "server": ALL_PAGES + SERVER_PAGES,
    "embedded": ALL_PAGES,
    "profiles": [p for p in ALL_PAGES if p[0] in ("history", "query-detail")],
}
# Running queries of the fixture's live list: each has a Cancel action.
RUNNING = sum(1 for q in fixture_server.DATA["live"] if q["state"] == "running")

# Anything cut at the screen's edge that no box of its own scrolls into view.
CUT = """
() => {
  const vw = document.documentElement.clientWidth, out = [];
  if (document.documentElement.scrollWidth > vw) out.push(`the page is ${document.documentElement.scrollWidth - vw}px wider than the screen`);
  const scrolls = e => {
    for (let a = e.parentElement; a && a !== document.body; a = a.parentElement) {
      const o = getComputedStyle(a).overflowX;
      if (o === 'auto' || o === 'scroll' || o === 'hidden') return true;
    }
    return false;
  };
  for (const e of document.querySelectorAll('body *')) {
    if (!e.checkVisibility({ visibilityProperty: true })) continue;
    const r = e.getBoundingClientRect();
    if (r.width <= 1 || r.height <= 1) continue;
    if ((r.right > vw + 1 || r.left < -1) && !scrolls(e)) {
      out.push(`${e.tagName.toLowerCase()} "${(e.textContent || '').trim().slice(0, 30)}" reaches ${Math.round(r.right)}px on a ${vw}px screen`);
      if (out.length > 3) break;
    }
  }
  return out;
}
"""

# A table's box: how far its content is wider than it, whether the box scrolls, and its width.
TABLES = """
() => [...document.querySelectorAll('[data-table-box]')]
  .filter(e => e.checkVisibility())
  .map(e => [e.scrollWidth - e.clientWidth, ['auto', 'scroll'].includes(getComputedStyle(e).overflowX), Math.round(e.getBoundingClientRect().width)])
"""

# Touch targets under 44 x 44 px. A link inside a sentence is text, not a control of its own.
SMALL_TARGETS = """
() => [...document.querySelectorAll('button, a, input, select, textarea, summary, [role="button"]')]
  .filter(e => e.checkVisibility({ visibilityProperty: true }))
  .map(e => [e, (e.tagName === 'INPUT' && e.closest('label') ? e.closest('label') : e).getBoundingClientRect()])
  .filter(([e, r]) => (r.height < 44 || r.width < 44) && !(e.tagName === 'A' && e.closest('p')))
  .map(([e, r]) => `${e.tagName.toLowerCase()} "${(e.textContent || e.getAttribute('aria-label') || '').trim().slice(0, 30)}" ${Math.round(r.width)}x${Math.round(r.height)}`)
"""

# The layout on a very wide screen: the content's box, the top bar's box, and the widest line of text and field
# (in units of their own font size).
WIDE = """
() => {
  const vw = document.documentElement.clientWidth;
  const box = s => { const r = document.querySelector(s).getBoundingClientRect(); return [Math.round(r.left), Math.round(vw - r.right), Math.round(r.width)]; };
  const em = e => e.getBoundingClientRect().width / parseFloat(getComputedStyle(e).fontSize);
  const widest = s => Math.round(Math.max(0, ...[...document.querySelectorAll(s)].filter(e => e.checkVisibility()).map(em)));
  return { main: box('main'), bar: box('header [data-shell]'), text: widest('main p'), field: widest('main input:not([type=checkbox]), main select') };
}
"""

MOTION = """
() => Math.max(0, ...[...document.querySelectorAll('*')].flatMap(e => {
  const s = getComputedStyle(e);
  return [...s.transitionDuration.split(','), ...s.animationDuration.split(',')].map(parseFloat);
}))
"""


def _open(
    browser: Browser, origin: str, size: tuple[str, int, int], touch: bool, **kw: object
) -> tuple[Page, list[str]]:
    _, width, height = size
    ctx = browser.new_context(
        viewport={"width": width, "height": height},
        has_touch=touch,
        is_mobile=touch and width < 500,
        **kw,
    )
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{origin}/?token=fixture")
    page.locator("header [data-shell]").wait_for(timeout=10000)
    page.wait_for_function("!location.search.includes('token')", timeout=10000)
    return page, errors


def _go(page: Page, frag: str, ready: str) -> bool:
    page.evaluate(f"window.location.hash = {frag!r}")
    try:
        # A page holds its table and its cards at once, one of them hidden: wait for the one on the screen.
        page.locator("main").get_by_text(ready, exact=False).locator(
            "visible=true"
        ).first.wait_for(timeout=8000)
    except PlaywrightTimeout:
        return False
    return True


def _layout(
    page: Page, view: str, name: str, mode: str, width: int, touch: bool
) -> list[str]:
    """The checks every page gets at every size."""
    problems = [f"{view}: {c}" for c in page.evaluate(CUT)]
    for over, scrolls, box in page.evaluate(TABLES):
        if over > 0 and width >= TABLES_FIT_FROM:
            problems.append(
                f"{view}: a table is {over}px wider than its box, so its last columns are cut"
            )
        elif over > 0 and not scrolls:
            problems.append(
                f"{view}: a table is {over}px wider than its box and the box does not scroll"
            )
        if box > 2000:
            problems.append(f"{view}: a table is {box}px wide (over 2,000)")
    if touch:
        problems += [
            f"{view}: touch target under 44 px: {s}"
            for s in page.evaluate(SMALL_TARGETS)
        ]
    if name == "queries" and mode != "profiles":
        # Each running query's Cancel is on the screen's width as it is: no sideways scroll of a table to reach it.
        shown = (
            page.locator("main")
            .get_by_role("button", name="Cancel")
            .locator("visible=true")
        )
        boxes = [shown.nth(i).bounding_box() for i in range(shown.count())]
        on_screen = sum(
            1
            for b in boxes
            if b is not None and b["x"] >= 0 and b["x"] + b["width"] <= width
        )
        if on_screen != RUNNING:
            problems.append(
                f"{view}: {on_screen} Cancel actions on the screen for {RUNNING} running queries"
            )
    if width >= ULTRAWIDE_FROM:
        w = page.evaluate(WIDE)
        left, right, across = w["main"]
        if abs(left - right) > 2:
            problems.append(
                f"{view}: the content is not centred ({left}px left, {right}px right)"
            )
        if not 1600 <= across <= 2000:
            problems.append(
                f"{view}: the content is {across}px wide (wanted 1,600 to 2,000)"
            )
        if w["bar"] != w["main"]:
            problems.append(
                f"{view}: the top bar ({w['bar']}) does not line up with the content ({w['main']})"
            )
        # About 90 characters of this typeface.
        if w["text"] > 50:
            problems.append(
                f"{view}: a paragraph is {w['text']} em wide (a readable line is under 50)"
            )
        if w["field"] > 50:
            problems.append(f"{view}: a field is {w['field']} em wide")
    return problems


# The long cell of the fixture's result (`fixture_server.RUNS`).
LONG_VALUE = "lorem ipsum " * 50


def becomes(page: Page, condition: str, timeout: int = 5000) -> bool:
    """Waits until the JavaScript expression `condition` is true in `page`; whether it was within `timeout` ms."""
    try:
        page.wait_for_function(f"() => {condition}", timeout=timeout)
    except PlaywrightTimeout:
        return False
    return True


def layout_checks(browser: Browser, shots: Path | None) -> list[str]:
    """Every page of every console at every size, dark and light."""
    problems: list[str] = []
    for mode, pages in MODE_PAGES.items():
        server = fixture_server.serve(mode=mode)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = f"http://127.0.0.1:{server.server_address[1]}"
        for scheme in ("light", "dark"):
            for sizes, touch in ((TOUCH, True), (MOUSE, False)):
                page, errors = _open(
                    browser, origin, sizes[0], touch, color_scheme=scheme
                )
                for name, frag, ready in pages:
                    for i, (label, width, height) in enumerate(sizes):
                        view = f"{mode} {name} {label} ({width}x{height}) {scheme}"
                        page.set_viewport_size({"width": width, "height": height})
                        if i == 0 and not _go(page, frag, ready):
                            problems.append(f"{view}: the page did not show {ready!r}")
                            break
                        page.wait_for_timeout(120)
                        problems += _layout(page, view, name, mode, width, touch)
                        if shots is not None and label in SHOT_SIZES:
                            folder = shots / mode
                            folder.mkdir(parents=True, exist_ok=True)
                            page.screenshot(
                                path=str(folder / f"{name}-{label}-{scheme}.png"),
                                full_page=True,
                            )
                problems += [f"{mode} {scheme}: page error: {e}" for e in errors]
                page.context.close()
        server.shutdown()
    return problems


def menu_checks(browser: Browser, origin: str) -> list[str]:
    """The menu of a narrow screen opens, keeps the focus inside while open, and closes on Esc and on a choice; a
    wider screen shows the pages in the top bar instead."""
    problems: list[str] = []
    for size in (TOUCH[0], TOUCH[1], MOUSE[0]):
        label = size[0]
        page, errors = _open(browser, origin, size, size in TOUCH)
        button = page.get_by_role("button", name="Open the menu")
        dialog = page.get_by_role("dialog")
        button.click()
        try:
            dialog.get_by_role("link", name="History").wait_for(timeout=3000)
        except PlaywrightTimeout:
            problems.append(f"menu {label}: the button does not open the pages")
            page.context.close()
            continue
        problems += (
            [
                f"menu {label}: touch target under 44 px: {s}"
                for s in page.evaluate(SMALL_TARGETS)
            ]
            if size in TOUCH
            else []
        )
        for _ in range(12):
            page.keyboard.press("Tab")
            if not page.evaluate("!!document.activeElement.closest('[role=dialog]')"):
                problems.append(f"menu {label}: Tab leaves the open menu")
                break
        page.keyboard.press("Escape")
        if not becomes(page, "!document.querySelector('[role=dialog]')"):
            problems.append(f"menu {label}: Esc does not close the menu")
        elif not becomes(
            page,
            "document.activeElement.getAttribute('aria-label') === 'Open the menu'",
        ):
            problems.append(
                f"menu {label}: the focus does not return to the menu button"
            )
        button.click()
        dialog.get_by_role("link", name="History").click()
        if not becomes(
            page,
            "!document.querySelector('[role=dialog]') && location.hash.includes('#/history')",
        ):
            problems.append(
                f"menu {label}: choosing a page does not close the menu and go there"
            )
        problems += [f"menu {label}: page error: {e}" for e in errors]
        page.context.close()
    for size in (TOUCH[2], TOUCH[3], TOUCH[4]):
        page, _ = _open(browser, origin, size, True)
        links = page.locator("header nav a").locator("visible=true")
        if (
            links.count() != 10
            or page.get_by_role("button", name="Open the menu").is_visible()
        ):
            problems.append(
                f"menu {size[0]}: the top bar shows {links.count()} of the 10 pages"
            )
        page.context.close()
    return problems


def editor_checks(browser: Browser, origin: str) -> list[str]:
    """The SQL editor on a phone: Run and Cancel stay on the screen under a long statement with the keyboard up,
    the result scrolls inside its own box, and a long value is cut short until it is tapped."""
    problems: list[str] = []
    # A phone with the on-screen keyboard up (it takes about half the height), upright and on its side.
    for label, width, height in (
        ("phone, keyboard up", 390, 420),
        ("phone on its side, keyboard up", 844, 200),
    ):
        page, errors = _open(browser, origin, (label, width, height), True)
        view = f"sql {label}"
        _go(page, "#/sql", "Run")
        editor = page.locator(".cm-content")
        editor.click()
        page.keyboard.press("Control+A")
        page.keyboard.type("SELECT wide\n" + "  , a_long_column_name\n" * 30)
        page.wait_for_timeout(200)
        for name in ("Run", "Cancel"):
            box = page.get_by_role("button", name=name, exact=True).bounding_box()
            if box is None or box["y"] < 0 or box["y"] + box["height"] > height:
                problems.append(f"{view}: the {name} button is off the screen ({box})")
        page.get_by_role("button", name="Run", exact=True).click()
        try:
            page.get_by_text("finished", exact=True).first.wait_for(timeout=5000)
        except PlaywrightTimeout:
            problems.append(f"{view}: the statement did not finish")
            page.context.close()
            continue
        problems += [f"{view}: {c}" for c in page.evaluate(CUT)]
        result = page.locator(
            "[data-table-box]", has=page.get_by_role("table", name="Result")
        )
        over = result.evaluate(
            "e => [e.scrollWidth - e.clientWidth, getComputedStyle(e).overflowX, getComputedStyle(e).overflowY]"
        )
        if over[0] <= 0 or over[1] != "auto" or over[2] != "auto":
            problems.append(
                f"{view}: the result does not scroll inside its own box ({over})"
            )
        long = result.get_by_role("button", name="Show the whole value").first
        if long.count() == 0:
            problems.append(f"{view}: a long value has no control to show it whole")
        else:
            long.tap()
            whole = result.get_by_role("button", name="Show less").first
            # The fixture's long cell, every word of it, then the control's own (unseen) name.
            if not becomes(
                page,
                "(() => { const b = document.querySelector('[data-table-box] button[aria-expanded=true]');"
                f" return !!b && b.textContent.replace(/\\s+/g, ' ').trim() === {LONG_VALUE + 'Show less'!r}; }})()",
            ):
                shown = whole.inner_text() if whole.count() else "(no control)"
                problems.append(
                    f"{view}: a tap does not show a long value whole ({len(shown)} characters shown)"
                )
        problems += [
            f"{view}: touch target under 44 px: {s}"
            for s in page.evaluate(SMALL_TARGETS)
        ]
        problems += [f"{view}: page error: {e}" for e in errors]
        page.context.close()
    return problems


def preference_checks(browser: Browser, origin: str) -> list[str]:
    """Text at twice its size, a browser's minimum font size, reduced motion and the two colour schemes."""
    problems: list[str] = []
    for label, style in (
        # Text zoom: every size in the page is in rem, so the root's size scales them all.
        ("text at 200%", "html { font-size: 200% !important }"),
        # A minimum font size, as the browser's setting forces it: no text under 20 px.
        ("minimum font size 20 px", "* { font-size: max(20px, 1em) !important }"),
    ):
        for size, touch in ((TOUCH[1], True), (MOUSE[1], False)):
            # The page's policy refuses a style a script adds, so the test's own style goes past it.
            page, errors = _open(browser, origin, size, touch, bypass_csp=True)
            page.add_style_tag(content=style)
            for name, frag, ready in ALL_PAGES + SERVER_PAGES:
                view = f"{label}, {name} {size[0]}"
                if not _go(page, frag, ready):
                    problems.append(f"{view}: the page did not show {ready!r}")
                    continue
                page.wait_for_timeout(150)
                problems += [f"{view}: {c}" for c in page.evaluate(CUT)]
            problems += [f"{label}: page error: {e}" for e in errors]
            page.context.close()
    # Reduced motion: nothing moves or fades for longer than a frame, the open menu included.
    page, _ = _open(browser, origin, TOUCH[1], True, reduced_motion="reduce")
    page.get_by_role("button", name="Open the menu").click()
    page.get_by_role("dialog").wait_for(timeout=3000)
    longest = page.evaluate(MOTION)
    if longest > 0.02:
        problems.append(f"reduced motion: something still moves for {longest} s")
    page.context.close()
    # The colour scheme follows the system: a dark page on a dark system, a light one on a light system.
    seen = {}
    for scheme in ("light", "dark"):
        page, _ = _open(browser, origin, MOUSE[1], False, color_scheme=scheme)
        rgb = page.evaluate(
            "getComputedStyle(document.body).backgroundColor.match(/[0-9.]+/g).slice(0, 3).map(Number)"
        )
        seen[scheme] = sum(rgb) / 3
        page.context.close()
    if not (seen["light"] > 200 and seen["dark"] < 60):
        problems.append(
            f"colour scheme: the page's background is {seen} (light should be light, dark dark)"
        )
    return problems


def responsive_checks(shots: Path | None = None) -> list[str]:
    """Every check of this file, as a list of problems (empty when the console passes)."""
    server = fixture_server.serve()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        problems = layout_checks(browser, shots)
        problems += menu_checks(browser, origin)
        problems += editor_checks(browser, origin)
        problems += preference_checks(browser, origin)
        browser.close()
    server.shutdown()
    return problems


if __name__ == "__main__":
    found = responsive_checks(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    sizes = len(TOUCH) + len(MOUSE)
    print(
        "\n".join(found)
        if found
        else f"ui responsive: ok (3 consoles x {sizes} screen sizes x dark/light, menu, SQL editor on a phone, text size, motion, colour scheme)"
    )
    sys.exit(1 if found else 0)
