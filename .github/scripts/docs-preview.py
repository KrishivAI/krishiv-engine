#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the Markdown pages under docs/ (or only the changed ones) to static HTML.

Usage: docs-preview.py OUT_DIR [CHANGED_FILE ...]
Without changed files every page is rendered. Links between pages are rewritten from `.md` to
`.html`. The output is a CI artifact for reviewers, not the published site.
"""

import html
import pathlib
import re
import sys

import markdown  # installed by the workflow: `uv run --with markdown==3.7`

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{font:16px/1.5 system-ui,sans-serif;max-width:50rem;margin:2rem auto;padding:0 1rem}}
pre{{overflow:auto;background:#f4f4f4;padding:.75rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:.25rem .5rem}}</style>
<p><a href="{up}index.html">docs home</a> &middot; <code>{path}</code></p>{body}</html>"""


def render(src: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    rel = src.relative_to(DOCS)
    text = src.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    body = re.sub(r'href="([^"#:]+)\.md(#[^"]*)?"', r'href="\1.html\2"', body)
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = html.escape(m.group(1)) if m else rel.stem
    dest = out_dir / rel.with_suffix(".html")
    dest.parent.mkdir(parents=True, exist_ok=True)
    up = "../" * (len(rel.parts) - 1)
    dest.write_text(
        PAGE.format(title=title, up=up, path=rel, body=body), encoding="utf-8"
    )
    return dest


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    out_dir = pathlib.Path(argv[0])
    changed = [ROOT / p for p in argv[1:]]
    pages = [
        p for p in changed if p.suffix == ".md" and DOCS in p.parents and p.exists()
    ]
    if not argv[1:]:
        pages = sorted(DOCS.rglob("*.md"))
    for p in pages:
        render(p, out_dir)
    # A page always needs the home page to link back to.
    if (DOCS / "index.md").exists():
        render(DOCS / "index.md", out_dir)
    print(f"rendered {len(pages)} page(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
