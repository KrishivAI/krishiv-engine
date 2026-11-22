#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record Spark's output column names for `variant_get` / `try_variant_get` / `:` path extraction (SQL-41).

Spark source: `VariantGet` in `sql/catalyst/.../expressions/variant/variantExpressions.scala` and
`toPrettySQL` / `usePrettyExpression` at each tag. Every value comes from running the query in an
`apache/spark:<ver>` container (`SELECT <expr>` unaliased), never from the source.

Usage (from the repo root; containers are only read from):

    python tests/scripts/gen_variant_get_column_names.py 4.3.0=p3-1-spark43 4.2.0=s8-sql7-spark420 \
        4.1.0=p31-cws-spark-4.1.0 4.0.0=p31-cws-spark-4.0.0

Writes `tests/fixtures/spark-<ver>/column-names-variant-get.json`. `--runner` runs inside the container.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

LIT = 'parse_json(\'{"a":1,"b":{"c":"x"}}\')'
CASES: list[tuple[str, str]] = []
for tag, fn in (("vg", "variant_get"), ("tvg", "try_variant_get")):
    for ty in ("int", "string", "double", "array<int>"):
        k = ty.replace("<", "_").replace(">", "")
        CASES.append((f"{tag}_col_{k}", f"SELECT {fn}(v, '$.a', '{ty}') FROM t"))
        CASES.append((f"{tag}_lit_{k}", f"SELECT {fn}({LIT}, '$.a', '{ty}')"))
    CASES.append((f"{tag}_col_2arg", f"SELECT {fn}(v, '$.a') FROM t"))
    CASES.append((f"{tag}_lit_2arg", f"SELECT {fn}({LIT}, '$.a')"))
CASES += [
    ("path_col", "SELECT v:a FROM t"),
    ("path_col_nested", "SELECT v:b.c FROM t"),
    ("path_col_two", "SELECT v:a:b FROM t"),
    ("path_col_bracket", "SELECT v:['a'] FROM t"),
    ("path_col_cast_int", "SELECT v:a::int FROM t"),
    ("path_col_cast_string", "SELECT v:a::string FROM t"),
    ("path_lit", f"SELECT {LIT}:a"),
    ("path_lit_cast_int", f"SELECT {LIT}:a::int"),
    ("path_lit_two", f"SELECT {LIT}:a:b"),
]
SETUP = ['CREATE OR REPLACE TEMPORARY VIEW t AS SELECT parse_json(\'{"a":1,"b":{"c":"x"}}\') AS v']
COL_CASES: list[tuple[str, str]] = []


def runner(setup: list[str], cases: list[list[str]]) -> None:
    """Runs inside the container (spark-submit); prints one JSON document."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    for s in setup:
        spark.sql(s)
    out = []
    for cid, sql in cases:
        rec = {"id": cid, "sql": sql}
        try:
            df = spark.sql(sql)
            rec["columns"] = df.columns
            rec["types"] = [f.dataType.simpleString() for f in df.schema.fields]
            rec["rows"] = [[None if v is None else str(v) for v in r] for r in df.collect()]
        except Exception as e:
            je = getattr(e, "_origin", None) or getattr(e, "java_exception", None)
            err = {}
            if je is not None:
                for g in ("getCondition", "getErrorClass"):
                    try:
                        err["class"] = str(getattr(je, g)())
                        break
                    except Exception:
                        continue
                try:
                    err["sqlState"] = str(je.getSqlState())
                    err["params"] = {str(k): str(v) for k, v in sorted(je.getMessageParameters().items())}
                except Exception:
                    pass
                err["message"] = str(je.getMessage()).split("\n")[0]
            else:
                err["message"] = str(e).split("\n")[0]
            rec["error"] = err
        out.append(rec)
    print("@@JSON@@" + json.dumps({"version": spark.version, "results": out}, ensure_ascii=True))


def main() -> None:
    if sys.argv[1] == "--runner":
        runner(json.loads(sys.argv[2]), json.loads(sys.argv[3]))
        return
    me = Path(__file__).resolve()
    root = me.parents[2]
    cases = [[a, b] for a, b in CASES + COL_CASES]
    for spec in sys.argv[1:]:
        ver, ctr = spec.split("=")
        subprocess.run(["docker", "cp", str(me), f"{ctr}:/tmp/gen_variant_get_names.py"], check=True)
        p = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                "PYTHONIOENCODING=utf-8",
                "-e",
                "LC_ALL=C.UTF-8",
                ctr,
                "/opt/spark/bin/spark-submit",
                "--conf",
                "spark.driver.extraJavaOptions=-Dfile.encoding=UTF-8",
                "/tmp/gen_variant_get_names.py",
                "--runner",
                json.dumps(SETUP, ensure_ascii=True),
                json.dumps(cases, ensure_ascii=True),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        line = next(x for x in p.stdout.splitlines() if x.startswith("@@JSON@@"))
        doc = json.loads(line[len("@@JSON@@") :])
        assert doc["version"] == ver, (doc["version"], ver)
        doc = {
            "format_version": 1,
            "generator": "tests/scripts/gen_variant_get_column_names.py",
            "spark_version": doc["version"],
            "spark_tag": f"v{ver}",
            "setup": SETUP,
            "results": doc["results"],
        }
        d = root / "tests/fixtures" / f"spark-{ver}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "column-names-variant-get.json").write_text(
            json.dumps(doc, indent=1, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(ver, len(doc["results"]), "cases")


if __name__ == "__main__":
    main()
