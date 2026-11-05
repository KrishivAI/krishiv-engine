#!/usr/bin/env python3
"""Generate the moment-aggregate window golden (queue row "Comet checks", float sweep group (c)).

Every value comes from a REAL Spark 4.2 run. Each case is one moment aggregate over a whole
partition (`fn(...) OVER (PARTITION BY g)`) on a small table of awkward doubles (NULLs, NaN,
infinities, -0.0, large values, one-row and all-NULL groups). Rows are `id` plus the value as
Python's `repr` (`nan`, `inf`, `None`); the file is `{"cases": [{"sql": ..., "rows": [...]}]}`.

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \\
      python tests/scripts/gen_moment_window_goldens.py --remote sc://127.0.0.1:15012 \\
      --out tests/fixtures/spark-4.2.0/moment-window-goldens.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_moment_window_goldens.py"
VIEW = (
    "CREATE TEMPORARY VIEW m AS SELECT * FROM VALUES "
    "(1, 1, 1.0D, 2.0D), (2, 1, 2.0D, 4.1D), (3, 1, 4.0D, 5.9D), (4, 1, CAST(NULL AS DOUBLE), 1.0D), "
    "(5, 2, 1.0D, 1.0D), "
    "(6, 3, CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE)), (7, 3, CAST(NULL AS DOUBLE), 2.0D), "
    "(8, 4, CAST('NaN' AS DOUBLE), 1.0D), (9, 4, 2.0D, 3.0D), (10, 4, 5.0D, 7.0D), "
    "(11, 5, CAST('Infinity' AS DOUBLE), 1.0D), (12, 5, 2.0D, 3.0D), "
    "(13, 6, -0.0D, 0.0D), (14, 6, 0.0D, -0.0D), (15, 6, -0.0D, 1.0D), "
    "(16, 7, 1.0E300D, 1.0E300D), (17, 7, -1.0E300D, 3.0D), (18, 7, 5.0D, -1.0E300D), "
    "(19, 8, 3.0D, 3.0D), (20, 8, 3.0D, 3.0D), (21, 8, 3.0D, 3.0D) AS m(id, g, x, y)"
)
ONE = ["skewness", "kurtosis", "var_samp", "var_pop", "stddev_samp", "stddev_pop", "variance", "stddev"]
TWO = [
    "corr",
    "covar_samp",
    "covar_pop",
    "regr_count",
    "regr_avgx",
    "regr_avgy",
    "regr_slope",
    "regr_intercept",
    "regr_r2",
    "regr_sxx",
    "regr_syy",
    "regr_sxy",
]


def sql(fn: str, args: str) -> str:
    return f"SELECT id, {fn}({args}) OVER (PARTITION BY g) FROM m ORDER BY id"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(a.remote).create()
    spark.sql(VIEW).collect()
    queries = [sql(f, "x") for f in ONE] + [sql(f, "y, x") for f in TWO]
    cases = []
    for q in queries:
        try:
            rows = [[str(r[0]), repr(r[1])] for r in spark.sql(q).collect()]
            cases.append({"sql": q, "rows": rows})
        except Exception as e:
            cases.append({"sql": q, "error": str(e).split("JVM stacktrace")[0].strip()[:200]})
    spark.stop()
    doc = {"format_version": FORMAT_VERSION, "generator": GENERATOR, "view": VIEW, "cases": cases}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", a.out, len(cases), "cases")


if __name__ == "__main__":
    main()
