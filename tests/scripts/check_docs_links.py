# SPDX-License-Identifier: Apache-2.0
"""Link check for docs/ (P3-T074): every relative Markdown link must resolve to a file, and a
`#anchor` into a Markdown page must match one of its headings. External links are not fetched
(the build needs no network). Exit 1 and a list on any broken link."""

import pathlib
import re
import sys

DOCS = pathlib.Path("docs")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^```.*?^```", re.S | re.M)


def slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, drop everything but letters, digits, spaces, `-` and `_`, spaces to `-`."""
    s = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return s.replace(" ", "-")


def anchors(page: pathlib.Path, cache: dict) -> set[str]:
    if page not in cache:
        text = FENCE.sub("", page.read_text(encoding="utf-8"))
        seen: dict[str, int] = {}
        found = set()
        for m in re.finditer(r"^#{1,6} (.+)$", text, re.M):
            base = slug(m.group(1))
            n = seen.get(base, 0)
            seen[base] = n + 1
            found.add(base if n == 0 else f"{base}-{n}")  # a repeated heading gets -1, -2, ...
        cache[page] = found
    return cache[page]


def main() -> int:
    bad, cache = [], {}
    for page in sorted(DOCS.rglob("*.md")):
        text = FENCE.sub("", page.read_text(encoding="utf-8"))
        for m in LINK.finditer(text):
            target = m.group(1)
            if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("//"):
                continue
            path, _, frag = target.partition("#")
            dest = (page.parent / path).resolve() if path else page.resolve()
            if not dest.exists():
                bad.append(f"{page}: {target}: no such file")
            elif frag and dest.suffix == ".md" and frag not in anchors(dest, cache):
                bad.append(f"{page}: {target}: no such heading")
    print("\n".join(bad) if bad else "docs links ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
