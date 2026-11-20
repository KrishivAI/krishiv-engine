# SPDX-License-Identifier: Apache-2.0
"""Merge the per-client smoke results into one markdown table (report-only).

Usage: report.py DIR [--strict]

Reads DIR/go.json and DIR/scala.json (whichever exist), writes DIR/summary.md and prints it.
The exit status is 0 unless --strict is given and a client did not pass every check, or a client
produced no results file at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CLIENTS = ("go", "scala")
MARK = {"pass": "pass", "fail": "FAIL", "error": "ERROR", None: "-"}


def load(directory: Path) -> dict[str, dict]:
    """The results file of each client that wrote one."""
    found = {}
    for client in CLIENTS:
        path = directory / f"{client}.json"
        if path.is_file():
            found[client] = json.loads(path.read_text())
    return found


def render(found: dict[str, dict]) -> str:
    """The markdown summary: totals per client, then one row per check."""
    lines = ["# Spark Connect client smoke runs (report-only)", ""]
    for client, doc in found.items():
        results = doc["results"]
        passed = sum(1 for r in results if r["status"] == "pass")
        lines.append(f"- **{client}** ({doc['version']}): {passed}/{len(results)} pass")
    missing = [c for c in CLIENTS if c not in found]
    lines.extend(
        f"- **{c}**: no results file (the run did not complete)" for c in missing
    )
    names: list[str] = []
    by_client: dict[str, dict[str, dict]] = {}
    for client, doc in found.items():
        by_client[client] = {r["name"]: r for r in doc["results"]}
        names.extend(n for n in by_client[client] if n not in names)
    header = ["check", *found]
    lines += ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in names:
        row = [name]
        for client in found:
            r = by_client[client].get(name)
            row.append(MARK[r["status"] if r else None])
        lines.append("| " + " | ".join(row) + " |")
    details = [
        f"- `{c}` `{n}`: {r['detail']}"
        for c, rs in by_client.items()
        for n, r in rs.items()
        if r["status"] != "pass" and r.get("detail")
    ]
    if details:
        lines += ["", "## Not passing", "", *details]
    return "\n".join(lines) + "\n"


def all_pass(found: dict[str, dict]) -> bool:
    """True when every expected client wrote results and each check passed."""
    return len(found) == len(CLIENTS) and all(
        r["status"] == "pass" for doc in found.values() for r in doc["results"]
    )


def main(argv: list[str]) -> int:
    """Entry point."""
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    directory = Path(args[0])
    found = load(directory)
    text = render(found)
    (directory / "summary.md").write_text(text)
    print(text)
    return 1 if "--strict" in argv and not all_pass(found) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
