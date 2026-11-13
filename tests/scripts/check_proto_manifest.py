#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify (offline) or write the vendored Spark Connect proto manifest (P3-T010).

`tests/connect/proto-manifest.json` records, for every vendored `proto/spark-connect/<ver>/spark/connect/*.proto`,
the upstream tag it was read from and its sha256. The check recomputes the hashes and fails on any drift. It
also enforces `version_delta_is_exactly_this`: the 4.2.1 -> 4.3 difference is limited to the known set of
files and added field names, so the next bump is read, not assumed.

    python3 tests/scripts/check_proto_manifest.py            # verify (PR check)
    python3 tests/scripts/check_proto_manifest.py --write    # regenerate after a deliberate re-vendor
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import sys
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    current = collect()
    if args.write:
        MANIFEST.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")
        return 0
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = [f"{v}: manifest differs from the vendored files" for v in current if current[v] != recorded.get(v)]
    errors += delta_problems()
    for e in errors:
        print(e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
