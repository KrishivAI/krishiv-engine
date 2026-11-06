#!/usr/bin/env python3
"""Generate the moment-aggregate window golden (queue row "Comet checks", float sweep group (c)).

Every value comes from a REAL Spark 4.2 run. Each case is one moment aggregate over a whole
partition, a peer-running frame, a ROWS running frame, a sliding frame or a frame ending after the
current row (`fn(...) OVER (PARTITION BY g ...)`) on a small table of awkward doubles (NULLs, NaN,
infinities, -0.0, large values, one-row and all-NULL groups). Rows are `id` plus the value as
Python's `repr` (`nan`, `inf`, `None`); the file is `{"cases": [{"sql": ..., "rows": [...]}]}`.

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \\
      python tests/scripts/gen_moment_window_goldens.py --remote sc://127.0.0.1:15012 \\
      --out tests/fixtures/spark-4.2.0/moment-window-goldens.json

or against a local Spark 4.2.0 (`krishiv-internal/ops/scripts/spark-local.sh`, which runs a script with PySpark
4.2.0 and a JDK): write a one-line wrapper that sets `sys.argv = [..., "--local", "--out", PATH]` and runs this file.

Besides the default block (ANSI on, every function and frame) there are three small blocks that carry a `conf`
object the test applies before running the case: the same one-argument and two-argument statistics with ANSI off
and with `spark.sql.legacy.statisticalAggregate=true` (two frames each), and `corr` values over the groups whose
`x` varies (the default block's `corr` only ever raises: group 6 has a constant `x`).
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


FRAMES = [
    "",
    "ORDER BY id",
    "ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
    "ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING",
    "ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING",
]


def sql(fn: str, args: str, frame: str) -> str:
    over = f"PARTITION BY g {frame}".strip()
    return f"SELECT id, {fn}({args}) OVER ({over}) FROM m ORDER BY id"


# The flag-sensitive statistics (a division by zero where Spark gives NULL, or NaN under the legacy flag).
FLAG_ONE = ["skewness", "kurtosis", "var_samp", "stddev_samp"]
FLAG_TWO = ["corr", "covar_samp"]
FLAG_FRAMES = ["", "ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"]
# Groups whose x varies (no constant x), so `corr` has values to compare.
VARYING = "WHERE g NOT IN (6, 8)"


def where_sql(fn: str, args: str, frame: str) -> str:
    over = f"PARTITION BY g {frame}".strip()
    return f"SELECT id, {fn}({args}) OVER ({over}) FROM m {VARYING} ORDER BY id"


def blocks() -> list[tuple[dict[str, str], list[str]]]:
    default = [sql(f, "x", fr) for fr in FRAMES for f in ONE]
    default += [sql(f, "y, x", fr) for fr in FRAMES for f in TWO]
    flagged = [sql(f, "x", fr) for fr in FLAG_FRAMES for f in FLAG_ONE]
    flagged += [sql(f, "y, x", fr) for fr in FLAG_FRAMES for f in FLAG_TWO]
    corr_values = [where_sql("corr", "y, x", fr) for fr in FRAMES]
    return [
        ({}, default),
        ({}, corr_values),
        ({"spark.sql.ansi.enabled": "false"}, flagged),
        ({"spark.sql.legacy.statisticalAggregate": "true"}, flagged),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote")
    ap.add_argument("--local", action="store_true", help="a local Spark session instead of Spark Connect")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pyspark.sql import SparkSession

    if a.local:
        spark = (
            SparkSession.builder.master("local[1]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.remote(a.remote).create()
    spark.sql(VIEW).collect()
    cases = []
    for conf, queries in blocks():
        for key, value in conf.items():
            spark.conf.set(key, value)
        for q in queries:
            case = {"sql": q}
            if conf:
                case["conf"] = conf
            try:
                case["rows"] = [[str(r[0]), repr(r[1])] for r in spark.sql(q).collect()]
            except Exception as e:
                case["error"] = str(e).split("JVM stacktrace")[0].strip()[:200]
            cases.append(case)
        for key in conf:
            spark.conf.unset(key)
    spark.stop()
    doc = {"format_version": FORMAT_VERSION, "generator": GENERATOR, "view": VIEW, "cases": cases}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", a.out, len(cases), "cases")


if __name__ == "__main__":
    main()
