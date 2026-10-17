#!/usr/bin/env python3
"""Generate the Spark 4.2.0 error-class fixtures (P2-T003, ADR 0008).

Nothing is transcribed: both outputs come from a REAL Spark 4.2.0 (the image pinned in
bench/baselines/docker-compose.yml, run as a one-off throw-away container):

1. `crates/krishiv-common/vendored/spark-error-conditions.json` is `error/error-conditions.json` read
   verbatim out of the `spark-common-utils` jar (Apache-2.0, see NOTICE). `build.rs` generates the
   registry from it.
2. `tests/fixtures/spark-4.2.0/error-classes/render.json` holds SAMPLES leaf classes whose parameters
   are filled with awkward values (quotes, `$`, backslashes, `<angle>` text, unicode, newlines). Each
   is raised as a real `SparkException(errorClass, params)` in `spark-shell`
   (`errclasses/render_errors.scala`); the fixture records Spark's `getSqlState` and `getMessage`.

Usage (from repo root; needs docker, no running Spark required):

    python3 tests/scripts/gen_error_classes.py

Output is deterministic. Pass `--container NAME` to reuse a running Spark 4.2.0 container instead.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JAR = "/opt/spark/jars/spark-common-utils_2.13-4.2.0.jar"
ENTRY = "error/error-conditions.json"
PLACEHOLDER = re.compile(r"<([a-zA-Z0-9_-]+)>")

# Sampled leaf classes. Includes classes with sub-classes, multi-line messages, several parameters,
# no parameters, no SQLSTATE, and the classes the engine raises first.
SAMPLES = [
    "DIVIDE_BY_ZERO",
    "ARITHMETIC_OVERFLOW",
    "CAST_INVALID_INPUT",
    "CAST_OVERFLOW",
    "INTERNAL_ERROR",
    "PARSE_SYNTAX_ERROR",
    "TABLE_OR_VIEW_NOT_FOUND",
    "UNRESOLVED_COLUMN.WITH_SUGGESTION",
    "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION",
    "DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES",
    "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    "AMBIGUOUS_ALIAS_IN_NESTED_CTE",
    "CANNOT_MERGE_SCHEMAS",
    "INVALID_FRACTION_OF_SECOND",
    "NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION",
    "INVALID_ARRAY_INDEX",
    "AGGREGATE_OUT_OF_MEMORY",
    "ALL_PARAMETERS_MUST_BE_NAMED",
    "COLUMN_ALREADY_EXISTS",
    "_LEGACY_ERROR_TEMP_3092",
]

VALUES = [
    "'abc'",
    '"STRING"',
    "`a`.`b`",
    "a$1 \\n ${x}",
    "<other> and <nested>",
    "line1\nline2",
    "naïve ✓ 日本",
    "",
]


def image() -> str:
    compose = (ROOT / "bench/baselines/docker-compose.yml").read_text(encoding="utf-8")
    m = re.search(r"image:\s*(apache/spark:4\.2\.0@sha256:[0-9a-f]{64})", compose)
    if not m:
        raise SystemExit("pinned Spark 4.2.0 image not found in docker-compose.yml")
    return m.group(1)


def run(cmd: list[str], stdin: str = "") -> str:
    done = subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=False, timeout=1800)
    if done.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed:\n{done.stderr[-2000:]}")
    return done.stdout


def leaves(doc: dict) -> dict[str, str]:
    """Leaf class name -> template (Spark: main + ' ' + sub-class message, lines joined by newline)."""
    out = {}
    for name, info in doc.items():
        main = "\n".join(info["message"])
        subs = info.get("subClass")
        if subs is None:
            out[name] = main
        else:
            for sub, sub_info in subs.items():
                out[f"{name}.{sub}"] = main + " " + "\n".join(sub_info["message"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", help="reuse this running Spark 4.2.0 container")
    args = ap.parse_args()
    if args.container:
        base = ["docker", "exec", "-i", args.container]
    else:
        base = ["docker", "run", "--rm", "-i", "--entrypoint", "", image()]

    extract = f"import zipfile,sys;sys.stdout.buffer.write(zipfile.ZipFile('{JAR}').read('{ENTRY}'))"
    raw = subprocess.run([*base, "python3", "-c", extract], capture_output=True, check=True).stdout
    doc = json.loads(raw)
    table = leaves(doc)
    (ROOT / "crates/krishiv-common/vendored/spark-error-conditions.json").write_bytes(raw)

    cases = []
    for i, name in enumerate(SAMPLES):
        if name not in table:
            raise SystemExit(f"sample {name} is not a 4.2.0 leaf class")
        params = {}
        for j, p in enumerate(dict.fromkeys(PLACEHOLDER.findall(table[name]))):
            params[p] = VALUES[(i + j) % len(VALUES)]
        cases.append({"c": name, "p": params})

    script = (Path(__file__).parent / "errclasses/render_errors.scala").read_text(encoding="utf-8")
    shell = (
        "cat > /tmp/krishiv_in.b64; base64 -d /tmp/krishiv_in.b64 > /tmp/krishiv_bundle; "
        "python3 - <<'PY'\nimport json\nb=json.load(open('/tmp/krishiv_bundle'))\n"
        "open('/tmp/krishiv_script.scala','w').write(b['s'])\n"
        "open('/tmp/krishiv_cases.jsonl','w').write(b['d'])\nPY\n"
        "/opt/spark/bin/spark-shell --master 'local[1]' -I /tmp/krishiv_script.scala 2>/dev/null"
    )
    import base64

    bundle = json.dumps({"s": script, "d": "".join(json.dumps(c) + "\n" for c in cases)})
    out = run([*base, "sh", "-c", shell], base64.b64encode(bundle.encode()).decode())
    lines = [ln for ln in out.splitlines() if ln.startswith("@@")]
    if "@@END" not in lines:
        raise SystemExit("spark-shell did not complete:\n" + out[-2000:])
    version = next(ln.split(" ", 1)[1] for ln in lines if ln.startswith("@@VERSION "))
    if version != "4.2.0":
        raise SystemExit(f"expected Spark 4.2.0, got {version}")
    results = [json.loads(ln[len("@@CASE ") :]) for ln in lines if ln.startswith("@@CASE ")]
    fixture = {
        "spark_version": version,
        "generator": "tests/scripts/gen_error_classes.py",
        "top_level_classes": len(doc),
        "leaf_classes": len(table),
        "cases": [
            {"class": r["c"], "sql_state": r["state"], "parameters": r["params"], "message": r["full"]} for r in results
        ],
    }
    dest = ROOT / "tests/fixtures/spark-4.2.0/error-classes"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "render.json").write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(doc)} top-level classes, {len(table)} leaves, {len(results)} rendered", file=sys.stderr)


if __name__ == "__main__":
    main()
