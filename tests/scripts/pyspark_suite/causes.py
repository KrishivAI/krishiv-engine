# SPDX-License-Identifier: Apache-2.0
"""Group the failures of a `run` by cause (error class, else exception type and a normalised first line)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

_CLASS = re.compile(r"\[([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\]")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b|#\d+|0x[0-9a-f]+", re.I)
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")


def cause_of(kind: str, message: str, text: str) -> str:
    """A short stable key: `[ERROR_CLASS]` when the server named one, else `Type: normalised first line`."""
    blob = f"{message}\n{text}"
    m = _CLASS.search(blob)
    if m and m.group(1) not in ("DEFAULT", "ERROR"):
        return m.group(1)
    first = next((ln.strip() for ln in message.splitlines() if ln.strip()), kind)
    first = _QUOTED.sub("'_'", first)
    first = _NUM.sub("N", first)
    return first[:110] if kind in ("failure", "error") else f"{kind}: {first[:110]}"


def collect(results: Path) -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    ids: dict[str, list[str]] = defaultdict(list)
    for xml in sorted(results.glob("*.xml")):
        rel = xml.stem.replace("__", "/")
        for case in ET.parse(xml).getroot().iter("testcase"):
            for child in case:
                if child.tag not in ("failure", "error"):
                    continue
                kind = (child.get("type") or child.tag).rsplit(".", 1)[-1]
                key = cause_of(kind, child.get("message") or "", child.text or "")
                counts[key] += 1
                cls = (case.get("classname") or "").rsplit(".", 1)[-1]
                ids[key].append(f"{rel}::{cls}::{case.get('name')}")
                break
    return counts, ids


def report(version: str, results: Path, top: int = 25) -> str:
    counts, ids = collect(results)
    total = sum(counts.values())
    lines = [
        f"## {version}: {total} failing tests, {len(counts)} causes",
        "",
        "| count | cause | example |",
        "|---:|---|---|",
    ]
    for key, n in counts.most_common(top):
        lines.append(f"| {n} | `{key.replace('|', '/')}` | `{ids[key][0]}` |")
    rest = sum(counts.values()) - sum(n for _, n in counts.most_common(top))
    if rest:
        lines.append(f"| {rest} | (other causes) | |")
    return "\n".join(lines) + "\n"
