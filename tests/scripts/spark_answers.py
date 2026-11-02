#!/usr/bin/env python3
"""Spark 4.2's answers to the statements of a reduced metamorphic finding (P2-T110b).

Reads {"data": DIR, "setup": [...], "queries": [...]} on stdin, registers the TPC-H Parquet files
under DIR as temporary views (`orders`, `customer`, ...), runs the setup statements, then each
query on a REAL Spark in local mode, and prints one answer per query as JSON:
{"rows": [[cell, ...], ...]} or {"error": "<error class or message>"}. Cells are text; NULL is
"NULL", NaN is "NaN", infinities "Infinity" / "-Infinity". Run by `metamorphic_reduce::PythonSpark`:

    env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \
        python -I tests/scripts/spark_answers.py < request.json
"""

import json
import math
import os
import sys


def cell(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
    return str(v)


def main():
    from pyspark.sql import SparkSession

    req = json.load(sys.stdin)
    spark = SparkSession.builder.master("local[2]").appName("krishiv-spark-answers").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    data = req.get("data")
    if data:
        for name in sorted(os.listdir(data)):
            path = os.path.join(data, name)
            if os.path.isdir(path) or name.endswith(".parquet"):
                spark.read.parquet(path).createOrReplaceTempView(name.removesuffix(".parquet"))
    for stmt in req["setup"]:
        spark.sql(stmt)
    out = []
    for q in req["queries"]:
        try:
            out.append({"rows": [[cell(c) for c in row] for row in spark.sql(q).collect()]})
        except Exception as e:  # noqa: BLE001 - the error text is the answer
            cls = getattr(e, "getErrorClass", lambda: None)()
            out.append({"error": cls or str(e).splitlines()[0]})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
