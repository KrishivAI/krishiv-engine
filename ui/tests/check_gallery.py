# SPDX-License-Identifier: Apache-2.0
"""The @krishiv/ui gallery under the console's production CSP: the policy check of the shared-UI note (section 4),
axe on every component page, and the behaviour of the floating parts.

    pnpm -C ui test                                   # builds ui/apps/gallery/dist after the unit tests
    uv run --with playwright python ui/tests/check_gallery.py

Serves the built gallery with the same `Content-Security-Policy` the engine sends (`fixture_server.CSP`, so
`style-src 'self'`: no style attribute, no `<style>` made at run time) and fails when a page reports a policy
violation or a console error, also after opening every popover kind (select, dialog, menu); when axe finds a
serious or critical violation, in dark and in light, with the popovers closed and open; when a floating layer
is not placed next to its trigger (Bits UI positions it through CSSOM: a blocked style would leave it in the
page's flow); or when the 10,000-row grid puts more than a window of rows in the page or loses its sticky header.
A self-test first serves a page with an inline style attribute and fails if the check does not see it.
"""

from __future__ import annotations

import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from fixture_server import CSP

DIST = Path(__file__).parent.parent / "apps" / "gallery" / "dist"
AXE = Path(__file__).parent.parent / "node_modules" / "axe-core" / "axe.min.js"
PAGES = [
    "button",
    "field",
    "select",
    "callout",
    "badge",
    "dialog",
    "tabs",
    "menu",
    "grid",
    "states",
]
SELFTEST = "/csp-selftest.html"
SELFTEST_HTML = b'<!doctype html><html lang="en"><title>t</title><p style="color: red">inline</p></html>'
# Recorded before any page script runs (an init script is not subject to the page's policy).
LISTEN = """
window.__csp = [];
document.addEventListener('securitypolicyviolation', e =>
  window.__csp.push(`${e.violatedDirective} blocked ${e.blockedURI || 'inline'} (${(e.sample || '').slice(0, 60)})`));
"""
AXE_RUN = (
    "axe.run().then(r => r.violations.filter(v => ['serious','critical'].includes(v.impact))"
    ".map(v => v.id + ': ' + v.nodes.slice(0, 2).map(n => n.target.join(' ')).join(' | ')))"
)


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Content-Security-Policy", CSP)
        super().end_headers()

    def do_GET(self) -> None:
        if self.path == SELFTEST:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(SELFTEST_HTML)))
            self.end_headers()
            self.wfile.write(SELFTEST_HTML)
            return
        super().do_GET()


def serve() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(DIST))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def violations(page: Page) -> list[str]:
    return page.evaluate("window.__csp")


def axe(page: Page, where: str) -> list[str]:
    page.evaluate(AXE.read_text())
    return [f"axe {where}: {v}" for v in page.evaluate(AXE_RUN)]


def near(page: Page, trigger: str, layer: str, where: str) -> list[str]:
    """The floating layer is placed (fixed or absolute, by Bits UI's CSSOM writes) and next to its trigger."""
    t = page.locator(trigger).first.bounding_box()
    box = page.locator(layer).first.bounding_box()
    position = page.locator(layer).first.evaluate(
        "e => getComputedStyle(e.closest('[data-bits-floating-content-wrapper]') || e).position"
    )
    if t is None or box is None:
        return [f"{where}: the layer or its trigger is not shown"]
    problems = []
    if position not in ("fixed", "absolute"):
        problems.append(
            f"{where}: the floating layer is not positioned ({position}): its styles were blocked"
        )
    if (
        abs(box["y"] - (t["y"] + t["height"])) > 40
        and abs(t["y"] - (box["y"] + box["height"])) > 40
    ):
        problems.append(
            f"{where}: the layer is not next to its trigger (trigger {t}, layer {box})"
        )
    return problems


def popovers(page: Page, origin: str, scheme: str) -> list[str]:
    """Opens every popover kind, uses it with the keyboard and the mouse, and runs axe while it is open."""
    problems: list[str] = []
    # Select: opens below its trigger, a click picks, the trigger shows the choice.
    page.goto(f"{origin}/#/select")
    trigger = "#sel-size"
    page.locator(trigger).click()
    page.get_by_role("listbox").wait_for(timeout=5000)
    problems += near(page, trigger, "[role=listbox]", f"select ({scheme})")
    problems += axe(page, f"#/select open ({scheme})")
    page.get_by_role("option", name="M, 16 vCPU, 64 GiB").click()
    page.get_by_role("listbox").wait_for(state="hidden", timeout=5000)
    if page.locator("[data-chosen] .mono").inner_text() != "m":
        problems.append(f"select ({scheme}): a click did not choose the option")
    if "M, 16 vCPU" not in page.locator(trigger).inner_text():
        problems.append(f"select ({scheme}): the trigger does not show the choice")
    # The keyboard: Enter opens, the arrows move, Enter picks; a disabled option is skipped.
    page.locator(trigger).focus()
    page.keyboard.press("Enter")
    page.get_by_role("listbox").wait_for(timeout=5000)
    # As for the menu: arrows count only once the open list has its highlighted option.
    page.locator("[role=listbox] [data-highlighted]").wait_for(timeout=5000)
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    page.get_by_role("listbox").wait_for(state="hidden", timeout=5000)
    if page.locator("[data-chosen] .mono").inner_text() == "xl":
        problems.append(f"select ({scheme}): the keyboard chose a disabled option")

    # Dialog: a modal with its title as its name; Escape closes it and the focus returns to the trigger.
    page.goto(f"{origin}/#/dialog")
    page.get_by_role("button", name="Cancel query…").click()
    dialog = page.get_by_role("dialog", name="Cancel Arjun Rao's query?")
    dialog.wait_for(timeout=5000)
    if (
        page.locator("[role=dialog]").first.evaluate(
            "e => getComputedStyle(e).position"
        )
        != "fixed"
    ):
        problems.append(f"dialog ({scheme}): not a fixed layer")
    problems += axe(page, f"#/dialog open ({scheme})")
    page.keyboard.press("Escape")
    dialog.wait_for(state="hidden", timeout=5000)
    focused = page.evaluate(
        "document.activeElement && document.activeElement.textContent"
    )
    if focused != "Cancel query…":
        problems.append(
            f"dialog ({scheme}): Escape did not return the focus to the trigger ({focused!r})"
        )
    page.get_by_role("button", name="Cancel query…").click()
    dialog.get_by_role("button", name="Cancel query").click()
    dialog.wait_for(state="hidden", timeout=5000)
    if page.locator("[data-outcome] .mono").inner_text() != "cancelled":
        problems.append(f"dialog ({scheme}): the primary button did not act")

    # Menu: opens below its trigger, the arrows move, Enter picks, a disabled item is skipped.
    page.goto(f"{origin}/#/menu")
    page.get_by_role("button", name="Actions").click()
    page.get_by_role("menu").wait_for(timeout=5000)
    problems += near(
        page, 'button:has-text("Actions")', "[role=menu]", f"menu ({scheme})"
    )
    problems += axe(page, f"#/menu open ({scheme})")
    page.keyboard.press("Escape")
    page.get_by_role("menu").wait_for(state="hidden", timeout=5000)
    page.get_by_role("button", name="Actions").focus()
    page.keyboard.press("Enter")
    page.get_by_role("menu").wait_for(timeout=5000)
    # The first item has the focus; Download results is disabled and skipped. Bits UI moves the focus into the
    # menu after it opens, so wait for it: arrows pressed before then are lost (seen on a CI runner).
    page.wait_for_function(
        "document.activeElement?.getAttribute('role') === 'menuitem'", timeout=5000
    )
    for _ in range(2):
        page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    page.get_by_role("menu").wait_for(state="hidden", timeout=5000)
    picked = page.locator("[data-picked] .mono").inner_text()
    if picked != "cancel":
        problems.append(
            f"menu ({scheme}): ArrowDown x2 + Enter picked {picked!r}, expected 'cancel' (disabled skipped)"
        )

    # Tabs: the arrows move between tabs and the panel follows.
    page.goto(f"{origin}/#/tabs")
    page.get_by_role("tab", name="Overview").click()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("Enter")
    if page.get_by_role("tabpanel").inner_text().strip() != "Sessions panel":
        problems.append(f"tabs ({scheme}): ArrowRight did not show the Sessions panel")
    return problems


def grid(page: Page, origin: str) -> list[str]:
    """10,000 rows: only a window of them is in the page, the scroll reaches the last row, the header stays."""
    problems: list[str] = []
    page.goto(f"{origin}/#/grid")
    box = page.get_by_role("region", name="Readings")
    box.locator("tbody tr").first.wait_for(timeout=5000)
    drawn = box.locator("tbody tr").count()
    if not 10 < drawn < 120:
        problems.append(
            f"grid: {drawn} rows in the page for 10,000 (expected a window)"
        )
    # The rows are as tall as the virtualizer counts them (the page sets rowHeight 40, not the default 36).
    heights = box.locator("tbody tr:not([aria-hidden])").evaluate_all(
        "rs => [...new Set(rs.map(r => Math.round(r.getBoundingClientRect().height)))]"
    )
    if heights != [40]:
        problems.append(f"grid: rows are {heights} px tall with rowHeight 40")
    box.evaluate("e => { e.scrollTop = e.scrollHeight; }")
    page.wait_for_timeout(300)
    if box.get_by_role("cell", name="10000", exact=True).count() != 1:
        problems.append("grid: scrolling to the end does not show row 10000")
    head = box.locator("thead").bounding_box()
    frame = box.bounding_box()
    if head is None or frame is None or abs(head["y"] - frame["y"]) > 2:
        problems.append(f"grid: the header is not sticky (header {head}, box {frame})")
    if box.locator("table").get_attribute("aria-rowcount") != "10001":
        problems.append("grid: aria-rowcount does not count every row")
    # Sorting: a click on a header sorts, aria-sort says so, a second click reverses.
    computes = page.get_by_role("region", name="Computes")
    computes.get_by_role("button", name="Compute").click()
    first = computes.locator("tbody tr").first.locator("td").first.inner_text()
    sort = computes.locator("th", has_text="Compute").get_attribute("aria-sort")
    if first != "bi" or sort != "ascending":
        problems.append(
            f"grid: sorting by Compute gives {first!r} first, aria-sort {sort!r}"
        )
    # Selection by keyboard: a selectable row takes focus and Enter selects it (WCAG 2.1.1).
    row = computes.locator("tbody tr", has_text="default")
    row.focus()
    page.keyboard.press("Enter")
    if row.get_attribute("aria-current") != "true":
        problems.append("grid: Enter on a focused row does not select it")
    # One tab stop for the rows (a roving tabindex), and the arrows move between them.
    stops = computes.locator("tbody tr[tabindex='0']").count()
    if stops != 1:
        problems.append(f"grid: {stops} rows are tab stops (expected 1)")
    page.keyboard.press("ArrowDown")
    after = page.evaluate("document.activeElement?.getAttribute('aria-rowindex')")
    if after != str(int(row.get_attribute("aria-rowindex") or "0") + 1):
        problems.append(f"grid: ArrowDown from the selected row focuses row {after}")
    # The 10,000-row grid: End reaches the last row even though it is not drawn yet.
    first_row = box.locator("tbody tr:not([aria-hidden])").first
    box.evaluate("e => { e.scrollTop = 0; }")
    page.wait_for_timeout(200)
    first_row.focus()
    page.keyboard.press("End")
    page.wait_for_timeout(300)
    if (
        page.evaluate("document.activeElement?.getAttribute('aria-rowindex')")
        != "10001"
    ):
        problems.append("grid: End on a virtual grid does not focus the last row")
    return problems


def support_id(page: Page, origin: str) -> list[str]:
    """The error state's support id is copyable (console-00 section 3, 4): the button copies it and says so."""
    problems: list[str] = []
    page.goto(f"{origin}/#/states")
    button = page.get_by_role("button", name="Copy support id")
    button.click()
    copied = page.evaluate("navigator.clipboard.readText()")
    if copied != "req_8c1f2a90":
        problems.append(
            f"states: the copy button put {copied!r} on the clipboard, not the support id"
        )
    live = page.locator("[aria-live=polite]", has_text="Copied")
    if live.count() != 1 or page.get_by_role("button", name="Copied").count() != 1:
        problems.append(
            "states: copying the support id does not show and announce Copied"
        )
    return problems


def gallery_checks() -> list[str]:
    if not (DIST / "index.html").exists():
        return [
            f"the gallery is not built ({DIST}): run `pnpm -C ui test` (or `pnpm -C ui --filter krishiv-ui-gallery build`)"
        ]
    if not AXE.exists():
        return [f"axe-core is not installed ({AXE}); run pnpm install in ui/"]
    problems: list[str] = []
    server = serve()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Self-test: the check must see an inline style the policy blocks, or it is vacuous.
        page = browser.new_page()
        page.add_init_script(LISTEN)
        page.goto(origin + SELFTEST)
        page.wait_for_timeout(200)
        if not any("style-src" in v for v in violations(page)):
            problems.append(
                "self-test: an inline style attribute was not reported as a CSP violation"
            )
        page.close()
        for scheme in ("dark", "light"):
            for width, height in ((390, 844), (1440, 900)):
                ctx = browser.new_context(
                    color_scheme=scheme,
                    viewport={"width": width, "height": height},
                    permissions=["clipboard-read", "clipboard-write"],
                )
                page = ctx.new_page()
                page.add_init_script(LISTEN)
                errors: list[str] = []
                page.on(
                    "console",
                    lambda m, errors=errors: (
                        errors.append(m.text) if m.type == "error" else None
                    ),
                )
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                view = f"{scheme} {width}px"
                for name in PAGES:
                    page.goto(f"{origin}/#/{name}")
                    page.locator("h1").first.wait_for(timeout=10000)
                    page.wait_for_timeout(150)
                    if page.evaluate(
                        "document.documentElement.scrollWidth > innerWidth + 1"
                    ):
                        problems.append(f"#/{name} ({view}): the page scrolls sideways")
                    if width == 1440:
                        problems += axe(page, f"#/{name} ({scheme})")
                if width == 1440:
                    problems += popovers(page, origin, scheme)
                    problems += grid(page, origin) if scheme == "dark" else []
                    problems += support_id(page, origin) if scheme == "dark" else []
                problems += [f"CSP ({view}): {v}" for v in violations(page)]
                problems += [f"console ({view}): {e}" for e in errors]
                ctx.close()
        browser.close()
    server.shutdown()
    return problems


if __name__ == "__main__":
    found = gallery_checks()
    print(
        "\n".join(found)
        if found
        else f"ui gallery: ok ({len(PAGES)} pages x dark/light x 390/1440 px, CSP style-src 'self' with select, "
        "dialog and menu open, axe closed and open, keyboard, 10,000-row virtual grid, support id copy)"
    )
    sys.exit(1 if found else 0)
