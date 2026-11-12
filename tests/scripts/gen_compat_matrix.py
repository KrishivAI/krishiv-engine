# SPDX-License-Identifier: Apache-2.0
"""Builds crates/krishiv-cli/src/check_compat/functions.json (T029) from the
generated SQL function pages. Run from the repo root; `--check` diffs only."""

import json
import pathlib
import re
import sys

root = pathlib.Path("docs/reference/sql/functions")
out = pathlib.Path("crates/krishiv-cli/src/check_compat/functions.json")
rows = {}
for page in sorted(root.glob("*.md")):
    if page.name == "index.md":
        continue
    text = page.read_text()
    m = re.search(r"^\| Krishiv \| (.+?) \|$", text, re.M)
    if not m:
        continue
    status = (
        "not yet" if m.group(1).startswith("Not implemented") else ("partial" if "(" in m.group(1) else "supported")
    )
    title = re.match(r"# `(.+?)`", text)
    rows[(title.group(1) if title else page.stem).lower()] = status
data = json.dumps(dict(sorted(rows.items())), indent=1) + "\n"
if "--check" in sys.argv:
    sys.exit(0 if out.exists() and out.read_text() == data else 1)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(data)
print(len(rows), "functions")
