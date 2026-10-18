#!/usr/bin/env python3
"""Generate the Spark built-in function catalog the function registry checks itself against (P2-T153).

Everything is produced by executing real Spark releases in throw-away containers; nothing is
transcribed from docs or memory:

1. Each supported release line's registry (`fnmatrix/dump_registry.scala`: functions, table functions
   and parser-level operators) gives the names it knows; a name's first release is the oldest image
   that lists it.
2. `fnregistry/function_traits.scala` analyzes up to `CASES_PER_FUNCTION` function-matrix cases per
   function on the newest release and reports `Expression.deterministic` and the call's kind.
3. `fnregistry/function_arity.scala` calls each function with 0 to 6 NULL arguments on the newest
   release: the counts that pass Spark's arity check, and Spark's `expectedNum` text for the others.

Output: `crates/krishiv-functions/vendored/spark-functions.tsv` (sorted; one row per name).

Usage (from the repo root; needs the images below and Docker):

    python3 tests/scripts/gen_function_registry.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGES = [
    ("3.5", "apache/spark:3.5.7"),
    ("4.0", "apache/spark:4.0.0"),
    ("4.1", "apache/spark:4.1.0"),
    ("4.2", "apache/spark:4.2.0"),
]
MATRIX = ROOT / "tests/fixtures/spark-4.2.0/function-matrix"
OUT = ROOT / "crates/krishiv-functions/vendored/spark-functions.tsv"
CASES_PER_FUNCTION = 8
MAX_ARGS = 6


def run_shell(image: str, script: str, data: str = "") -> list[str]:
    """Runs a Scala script in `spark-shell` in a throw-away container; returns its `@@` lines."""
    cid = subprocess.run(
        [
            "docker",
            "create",
            "-i",
            image,
            "/opt/spark/bin/spark-shell",
            "--master",
            "local[1]",
            "--driver-memory",
            "2g",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    try:
        for name, text in (("krishiv_script.scala", script), ("krishiv_cases.jsonl", data)):
            tmp = ROOT / "target" / f"fnregistry-{name}"
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_text(text, encoding="utf-8")
            subprocess.run(["docker", "cp", str(tmp), f"{cid}:/tmp/{name}"], check=True)
            tmp.unlink()
        # `:load` over stdin: Scala 2.13 REPLs without a terminal skip `-I` scripts.
        out = subprocess.run(
            ["docker", "start", "-a", "-i", cid],
            input=":load /tmp/krishiv_script.scala\n",
            capture_output=True,
            text=True,
            timeout=7200,
        ).stdout
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)
    lines = [line for line in out.splitlines() if line.startswith("@@")]
    if "@@END" not in lines:
        raise SystemExit(f"spark-shell on {image} did not complete:\n" + out[-2000:])
    return lines


def registry(image: str) -> dict[str, str]:
    script = (ROOT / "tests/scripts/fnmatrix/dump_registry.scala").read_text(encoding="utf-8")
    rows = [json.loads(line[5:]) for line in run_shell(image, script) if line.startswith("@@FN ")]
    return {r["name"]: r["registry"] for r in rows}


def matrix_cases() -> str:
    manifest = json.loads((MATRIX / "manifest.json").read_text(encoding="utf-8"))
    base = dict(manifest["base_conf"]) | manifest["modes"]["ansi_on"]
    per: dict[str, int] = defaultdict(int)
    out = []
    for fam in manifest["families"]:
        for line in (MATRIX / fam["shard"]).read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            f = case["function"]
            if per[f] >= CASES_PER_FUNCTION or "error" in (case.get("ansi_on") or {}):
                continue
            per[f] += 1
            out.append(json.dumps({"f": f, "sql": case["sql"], "conf": base | case.get("conf", {})}))
    return "\n".join(out) + "\n"


def main() -> int:
    first: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for line, image in IMAGES:
        for name, reg in registry(image).items():
            first.setdefault(name, line)
            kinds[name] = reg
    script = (ROOT / "tests/scripts/fnregistry/function_traits.scala").read_text(encoding="utf-8")
    seen: dict[str, set[tuple[bool, str]]] = defaultdict(set)
    for line in run_shell(IMAGES[-1][1], script, matrix_cases()):
        if line.startswith("@@T "):
            _, f, det, kind = line.split(" ")
            seen[f].add((det == "true", kind))
    script = (ROOT / "tests/scripts/fnregistry/function_arity.scala").read_text(encoding="utf-8")
    arity: dict[str, list[str]] = defaultdict(lambda: ["?"] * (MAX_ARGS + 1))
    for line in run_shell(IMAGES[-1][1], script, "\n".join(sorted(first)) + "\n"):
        if line.startswith("@@A "):
            _, f, n, verdict = line.split(" ", 3)
            arity[f][int(n)] = verdict.replace("_", " ")
    header = "name\tregistry\tsince\tdeterministic\tkind\targ_counts\texpected_num"
    rows = [f"# generated by tests/scripts/gen_function_registry.py; columns: {header}"]
    for name in sorted(first):
        traits = seen.get(name, set())
        det = "unknown" if not traits else str(all(d for d, _ in traits)).lower()
        kinds_seen = {k for _, k in traits}
        kind = kinds_seen.pop() if len(kinds_seen) == 1 else "unknown"
        verdicts = arity[name]
        counts = "".join("1" if v == "ok" else "0" for v in verdicts)
        expected = next((v for v in verdicts if v not in ("ok", "?")), "")
        rows.append(f"{name}\t{kinds[name]}\t{first[name]}\t{det}\t{kind}\t{counts}\t{expected}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(first)} names written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
