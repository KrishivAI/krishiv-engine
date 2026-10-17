#!/usr/bin/env python3
"""Write the seed corpus of the `sql_parse` fuzz target.

One file per statement of the vendored Spark goldens (`tests/spark-golden/**/*.sql.out`, the `-- !query`
blocks), each prefixed with the option byte the target reads first (0: default options), plus seeds of
the shapes that once cost super-linear time or memory (a `FROM` clause shared by many `SELECT` bodies,
nested calls, unterminated quotes).

Usage (from repo root): `python3 tests/scripts/fuzz_seed_sql.py fuzz/corpus/sql_parse`
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def statements(root: Path) -> list[str]:
    out = []
    for path in sorted(root.rglob("*.sql.out")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for block in text.split("-- !query\n")[1:]:
            out.append(block.split("\n-- !query schema", 1)[0])
    return out


def shapes() -> list[str]:
    tables = ", ".join(f"t{i}" for i in range(2000))
    return [
        f"FROM {tables} " + "SELECT 1 " * 2000,
        "SELECT " + "trim(" * 200 + "x" + ")" * 200,
        "SELECT " + "(" * 500 + "SELECT 1",
        "SELECT " + "'\\" * 5000,
    ]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fuzz/corpus/sql_parse")
    out.mkdir(parents=True, exist_ok=True)
    seeds = statements(Path("tests/spark-golden")) + shapes()
    for sql in seeds:
        data = b"\x00" + sql.encode("utf-8")
        (out / hashlib.sha1(data).hexdigest()[:16]).write_bytes(data)
    print(f"wrote {len(seeds)} seeds -> {out}")


if __name__ == "__main__":
    main()
