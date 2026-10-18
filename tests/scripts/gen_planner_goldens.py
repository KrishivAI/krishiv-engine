#!/usr/bin/env python3
"""Generate Spark 4.2.0's results for the physical planner's end-to-end tests (P2-T074).

Runs a REAL Spark (never from memory) in a throw-away container. Every `<name>.parquet` file of the data
directory becomes a temporary view `<name>`; every query of the query file runs and its rows are recorded
in order, each value rendered as text: decimals as plain numbers with their scale, dates as `YYYY-MM-DD`,
booleans as `true`/`false`, NULL as `NULL`; cells are joined by `|`. A query that fails records Spark's
error class instead. Usage (from repo root):

    docker run --rm -v "$PWD:/work:ro" apache/spark:4.2.0 \\
        /opt/spark/bin/spark-submit --master 'local[2]' /work/tests/scripts/gen_planner_goldens.py \\
        /work/<data dir> /work/<queries.json> [conf] | sed -n 's/^KRISHIV-GOLDEN //p' > <answers.json>

`queries.json` maps names to SQL text, or is `tpch`: the 22 queries of `bench/queries/tpch`. Optional
`conf` (JSON object) sets session configuration first. The fixture goes to stdout (the container user
cannot write the mount) as the line marked `KRISHIV-GOLDEN `: Spark also logs query errors to stdout.
"""

import datetime
import decimal
import json
import os
import sys

from pyspark.sql import SparkSession


MARK = "KRISHIV-GOLDEN "


def render(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, decimal.Decimal):
        return format(v, "f")
    if isinstance(v, datetime.datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, datetime.date):
        return v.isoformat()
    return str(v)


def error_class(e: Exception) -> str:
    """Spark's error condition: of the exception, or of the first cause in a job failure's chain."""
    cond = getattr(e, "getCondition", lambda: None)()
    if cond:
        return cond
    java = getattr(e, "java_exception", None)
    while java is not None:
        try:
            cond = java.getCondition()
        except Exception:  # noqa: BLE001 - a Java exception without conditions
            cond = None
        if cond:
            return cond
        java = java.getCause()
    return type(e).__name__


def main() -> None:
    data, queries = sys.argv[1], sys.argv[2]
    conf = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    spark = SparkSession.builder.appName("planner-goldens").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    for k, v in conf.items():
        spark.conf.set(k, v)
    for f in sorted(os.listdir(data)):
        if f.endswith(".parquet"):
            spark.read.parquet(os.path.join(data, f)).createOrReplaceTempView(f[: -len(".parquet")])
    if queries == "tpch":
        root = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "queries", "tpch")
        named = {f"q{q:02}": open(os.path.join(root, f"q{q:02}.sql")).read() for q in range(1, 23)}
    else:
        with open(queries) as f:
            named = json.load(f)
    out = {}
    for name, sql in named.items():
        try:
            rows = spark.sql(sql).collect()
            out[name] = {"rows": ["|".join(render(v) for v in r) for r in rows]}
        except Exception as e:  # noqa: BLE001 - Spark's error class is the expectation
            cls = error_class(e)
            out[name] = {"error": cls}
    # Spark logs query errors to stdout too: the fixture is the one marked line.
    fixture = json.dumps({"spark_version": spark.version, "conf": conf, "queries": out}, sort_keys=True)
    sys.stdout.write(f"{MARK}{fixture}\n")


if __name__ == "__main__":
    main()
