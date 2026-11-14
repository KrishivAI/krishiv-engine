#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify (offline) or write the vendored Spark Connect proto manifest (P3-T010).

`tests/connect/proto-manifest.json` records, for every vendored `proto/spark-connect/<ver>/spark/connect/*.proto`,
the upstream tag it was read from and its sha256. The check recomputes the hashes and fails on any drift. It
also enforces `version_delta_is_exactly_this`: the 4.2.1 -> 4.3 difference is limited to the known set of
files and added field names, so the next bump is read, not assumed.

    python3 tests/scripts/check_proto_manifest.py            # verify (PR check)
    python3 tests/scripts/check_proto_manifest.py --write    # regenerate after a deliberate re-vendor
    python3 tests/scripts/check_proto_manifest.py --upstream # nightly: re-read every tag from apache/spark and compare
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import itertools
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "proto" / "spark-connect"
MANIFEST = ROOT / "tests" / "connect" / "proto-manifest.json"
# vendored directory -> upstream apache/spark tag
TAGS = {"3.5": "v3.5.0", "4.0": "v4.0.0", "4.1": "v4.1.0", "4.2": "v4.2.0", "4.2.1": "v4.2.1", "4.3": "v4.3.0"}
# 4.2.1 -> 4.3.0: for each file that may differ, the exact statement lines (comments, blank lines and lone braces
# aside) the diff must add, with their counts. Nothing may be removed or changed, and nothing may be missing.
DELTA_ADDED = {
    "expressions.proto": {
        "TimestampNTZNanos timestamp_ntz_nanos = 29;": 1,
        "TimestampLTZNanos timestamp_ltz_nanos = 30;": 1,
        "message TimestampNTZNanos {": 1,
        "message TimestampLTZNanos {": 1,
        "int64 epoch_micros = 1;": 2,
        "int32 nanos_within_micro = 2;": 2,
        "optional int32 precision = 3;": 2,
    },
    "pipelines.proto": {
        "repeated Expression track_history_column_list = 11;": 1,
        "repeated Expression track_history_except_column_list = 12;": 1,
        "SCD_TYPE_2 = 2;": 1,
    },
    "relations.proto": {
        "Zip zip = 48;": 1,
        "message Zip {": 1,
        "Relation left = 1;": 1,
        "Relation right = 2;": 1,
    },
    "types.proto": {
        "TimestampNTZNanos timestamp_ntz_nanos = 29;": 1,
        "TimestampLTZNanos timestamp_ltz_nanos = 30;": 1,
        "message TimestampNTZNanos {": 1,
        "message TimestampLTZNanos {": 1,
        "optional int32 precision = 1;": 2,
        "uint32 type_variation_reference = 2;": 2,
    },
}
DELTA_FILES = set(DELTA_ADDED)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    on_disk = sorted(d.name for d in PROTO.iterdir() if d.is_dir())
    if on_disk != sorted(TAGS):
        raise SystemExit(f"vendored versions {on_disk} differ from TAGS {sorted(TAGS)}")
    out = {}
    for ver, tag in TAGS.items():
        files = sorted((PROTO / ver / "spark" / "connect").glob("*.proto"))
        out[ver] = {"tag": tag, "files": {f.name: sha256(f) for f in files}}
    return out


def _statement(line: str) -> str | None:
    """The line without indentation, or None for blanks, comments and lone braces."""
    t = line.strip()
    if not t or t.startswith(("//", "*", "/*")) or t in {"}", "};"}:
        return None
    return t


def delta_problems(base: str = "4.2.1", new: str = "4.3") -> list[str]:
    problems = []
    a, b = PROTO / base / "spark" / "connect", PROTO / new / "spark" / "connect"
    if sorted(p.name for p in a.glob("*.proto")) != sorted(p.name for p in b.glob("*.proto")):
        problems.append("the set of .proto files differs")
    for f in sorted(b.glob("*.proto")):
        old = a / f.name
        if old.exists() and sha256(old) != sha256(f) and f.name not in DELTA_FILES:
            problems.append(f"{f.name}: changed outside the known delta")
    for name, expected in sorted(DELTA_ADDED.items()):
        old_lines = (a / name).read_text().splitlines()
        new_lines = (b / name).read_text().splitlines()
        added: collections.Counter[str] = collections.Counter()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            for line in old_lines[i1:i2]:
                if _statement(line) is not None:
                    problems.append(f"{name}: removed or changed line {line.strip()!r}")
            added.update(st for st in map(_statement, new_lines[j1:j2]) if st is not None)
        for st in sorted(added.keys() | expected.keys()):
            if added[st] > expected.get(st, 0):
                problems.append(f"{name}: unexpected added line {st!r} (x{added[st] - expected.get(st, 0)})")
            elif added[st] < expected[st]:
                problems.append(f"{name}: missing added line {st!r} (x{expected[st] - added[st]})")
    return problems


# Items an older version had that a newer one removed (the only non-additive changes between consecutive versions),
# each with its reason. `LEGACY_3_5_FIELDS` in crates/krishiv-connect/src/wire.rs lets a 3.5 client still send them.
ALLOWED_REMOVALS = {
    ("3.5", "4.0"): {
        ("relations.proto", "CachedLocalRelation", 1),  # userId: the server must use the active user
        ("relations.proto", "CachedLocalRelation", 2),  # sessionId: likewise
    },
}


def _items(ver: str) -> dict:
    """(file, message path, field number) -> (label, type, name), and (file, enum path, "enum", number) -> name."""
    out: dict = {}
    for f in sorted((PROTO / ver / "spark" / "connect").glob("*.proto")):
        stack: list[tuple[str, str]] = []
        text = re.sub(r"/\*.*?\*/", "", f.read_text(), flags=re.S)
        for raw in text.splitlines():
            line = raw.split("//")[0].strip()
            if not line:
                continue
            if m := re.match(r"(message|enum|oneof)\s+(\w+)\s*\{", line):
                if line.endswith("}"):  # `message Unknown {}` on one line
                    continue
                stack.append((m.group(1), m.group(2)))
                continue
            if line.startswith("}"):
                if stack:
                    stack.pop()
                continue
            path = ".".join(n for k, n in stack if k != "oneof")
            kind = stack[-1][0] if stack else ""
            if kind in ("message", "oneof") and (
                m := re.match(r"(?:(optional|repeated)\s+)?([\w.<>, ]+?)\s+(\w+)\s*=\s*(\d+)", line)
            ):
                out[(f.name, path, int(m.group(4)))] = (m.group(1) or "", m.group(2).replace(" ", ""), m.group(3))
            elif kind == "enum" and (m := re.match(r"(\w+)\s*=\s*(-?\d+)\s*;", line)):
                out[(f.name, path, "enum", int(m.group(2)))] = m.group(1)
    return out


def additivity_problems(chain=("3.5", "4.0", "4.1", "4.2", "4.2.1", "4.3")) -> list[str]:
    """Each version must keep every field and enum value of the one before (label `optional` may be added),
    except the listed removals: one superset decoder is only sound if newer versions only add."""
    problems = []
    for old, new in itertools.pairwise(chain):
        a, b = _items(old), _items(new)
        allowed = ALLOWED_REMOVALS.get((old, new), set())
        for key, val in a.items():
            if key in allowed:
                if key in b:
                    problems.append(f"{old} -> {new}: {key} is listed as removed but still exists")
            elif key not in b:
                problems.append(f"{old} -> {new}: removed {key} {val}")
            elif b[key] != val and not (isinstance(val, tuple) and val[1:] == b[key][1:] and val[0] == ""):
                problems.append(f"{old} -> {new}: changed {key} {val} -> {b[key]}")
    return problems


# Spark 3.5 kept the protos under connector/connect; 4.0 moved them to sql/connect.
RAW = "https://raw.githubusercontent.com/apache/spark/{tag}/{base}/common/src/main/protobuf/spark/connect/{name}"


def upstream_problems(fetch=None) -> list[str]:
    """Each vendored file against the same file at its apache/spark tag (network; the nightly check)."""

    def default_fetch(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()

    fetch = fetch or default_fetch
    problems = []
    for ver, tag in TAGS.items():
        for f in sorted((PROTO / ver / "spark" / "connect").glob("*.proto")):
            try:
                body = fetch(RAW.format(tag=tag, name=f.name, base="connector/connect" if ver == "3.5" else "sql/connect"))
            except OSError as e:
                problems.append(f"{ver}/{f.name}: cannot fetch {tag}: {e}")
                continue
            if hashlib.sha256(body).hexdigest() != sha256(f):
                problems.append(f"{ver}/{f.name}: differs from apache/spark {tag}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--upstream", action="store_true", help="compare with apache/spark at each tag (network)")
    args = ap.parse_args(argv)
    if args.upstream:
        problems = upstream_problems()
        for e in problems:
            print(e, file=sys.stderr)
        return 1 if problems else 0
    current = collect()
    if args.write:
        MANIFEST.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")
        return 0
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = [f"{v}: manifest differs from the vendored files" for v in current if current[v] != recorded.get(v)]
    errors += delta_problems()
    errors += additivity_problems()
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
