#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump Spark 4.2.0's own function documentation (usage, examples, group, since, deprecation note).

Everything comes from `ExpressionInfo` of an executed Spark release (the same dump the function
catalog uses); nothing is transcribed. Output: `crates/krishiv-functions/vendored/spark-function-docs.jsonl`
(sorted by name; one JSON object per line). `cargo xtask docs-sql` reads it.

Usage (repo root; needs Docker and the image below):  python3 tests/scripts/gen_function_docs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_function_registry import ROOT, run_shell

IMAGE = "apache/spark:4.2.0"
OUT = ROOT / "crates/krishiv-functions/vendored/spark-function-docs.jsonl"


def main() -> int:
    script = (ROOT / "tests/scripts/fnmatrix/dump_registry.scala").read_text(encoding="utf-8")
    rows = {}
    for line in run_shell(IMAGE, script):
        if line.startswith("@@FN "):
            r = json.loads(line[5:])
            # A name in both registries (explode, stack, ...) documents the function: the
            # table-function registry is dumped last and must not overwrite it.
            if r["name"] in rows and r["registry"] == "table_function":
                continue
            rows[r["name"]] = {
                k: r[k] for k in ("name", "registry", "group", "since", "deprecated", "usage", "examples")
            }
    OUT.write_text("".join(json.dumps(rows[n], ensure_ascii=False) + "\n" for n in sorted(rows)), encoding="utf-8")
    print(f"wrote {len(rows)} functions to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
