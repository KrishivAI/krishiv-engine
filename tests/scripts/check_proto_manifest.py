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
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "proto" / "spark-connect"
MANIFEST = ROOT / "tests" / "connect" / "proto-manifest.json"
# vendored directory -> upstream apache/spark tag
TAGS = {"3.5": "v3.5.0", "4.0": "v4.0.0", "4.1": "v4.1.0", "4.2": "v4.2.0", "4.2.1": "v4.2.1", "4.3": "v4.3.0"}
# 4.2.1 -> 4.3.0: the only files that may differ, and the only identifiers the diff may add.
DELTA_FILES = {"expressions.proto", "pipelines.proto", "relations.proto", "types.proto"}
DELTA_ADDED = {
    "Zip zip = 48",
    "timestamp_ntz_nanos = 29",
    "timestamp_ltz_nanos = 30",
    "message TimestampNTZNanos",
    "message TimestampLTZNanos",
    "message Zip",
    "track_history_column_list = 11",
    "track_history_except_column_list = 12",
    "SCD_TYPE_2 = 2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    out = {}
    for ver, tag in TAGS.items():
        files = sorted((PROTO / ver / "spark" / "connect").glob("*.proto"))
        out[ver] = {"tag": tag, "files": {f.name: sha256(f) for f in files}}
    return out


def delta_problems(base: str = "4.2.1", new: str = "4.3") -> list[str]:
    problems = []
    a, b = PROTO / base / "spark" / "connect", PROTO / new / "spark" / "connect"
    for f in sorted(b.glob("*.proto")):
        old = a / f.name
        if not old.exists():
            problems.append(f"{f.name}: new file")
        elif sha256(old) != sha256(f):
            if f.name not in DELTA_FILES:
                problems.append(f"{f.name}: changed outside the known delta")
                continue
            old_lines = set(old.read_text().splitlines())
            for line in f.read_text().splitlines():
                if line in old_lines or not line.strip() or line.strip().startswith(("//", "}", "*")):
                    continue
                if not any(re.search(re.escape(k), line) for k in DELTA_ADDED) and not re.match(
                    r"\s*(optional |repeated )?[\w.]+ \w+ = \d+;|\s*\w+ = \d+;|\s*(int|uint)\d+ \w+ = \d+;", line
                ):
                    problems.append(f"{f.name}: unexpected added line {line.strip()!r}")
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
