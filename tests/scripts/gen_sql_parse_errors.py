#!/usr/bin/env python3
"""Generate the Spark SQL syntax-error fixture (P2-T077).

Every expectation comes from a REAL Spark 4.2 run of the statement below, through a Spark Connect
server (default session configs): whether the statement parses, and for a syntax error its error
condition, message parameters (the offending token and the recovery hint) and position. The cases
are where ANTLR's inline error recovery shows or withholds its `extra input` / `missing` hints:
closing tokens after a complete element, comma lists, casts, subscripts, subqueries, and clause
keywords at the end of the input.

Usage (needs `pyspark-client==4.2.0` and a Spark 4.2 Connect server):

    python3 tests/scripts/gen_sql_parse_errors.py sc://HOST:PORT \
      --out tests/fixtures/spark-4.2.0/sql-parse-errors.json
"""

from __future__ import annotations

import argparse
import json
import re

from pyspark.errors import PySparkException
from pyspark.sql import SparkSession

CASES = [
    # a subquery is committed to by prediction: the closing `)` is missing
    "SELECT (SELECT 1",
    "SELECT (SELECT 1 2)",
    "SELECT (SELECT 1) 2 3",
    "SELECT * FROM (SELECT 1",
    "SELECT * FROM (SELECT 1 x y)",
    "SELECT 1 FROM t WHERE a IN (SELECT 1",
    "SELECT a FROM t WHERE a IN (SELECT 1 FROM u",
    # a parenthesised expression is still being predicted: no hint
    "SELECT (1",
    "SELECT (1 2)",
    "SELECT (1 + 2",
    "SELECT (1 + 2 3)",
    "SELECT ((1)",
    "SELECT ((SELECT 1)",
    "SELECT (SELECT 1) +",
    # rows and IN lists: a second element makes the end of input extra
    "SELECT (1, 2",
    "SELECT (1, 2 3)",
    "SELECT x IN (1",
    "SELECT x IN (1 2)",
    "SELECT x IN (1, 2",
    "SELECT a FROM t WHERE a IN (1, 2",
    # function calls
    "SELECT f(a",
    "SELECT f(a b)",
    "SELECT f(a,",
    "SELECT f(a, b",
    "SELECT f(a, b c)",
    "SELECT f(g(a)",
    "SELECT f(g(a) b)",
    "SELECT f(DISTINCT a",
    "SELECT count(*",
    "SELECT count(a b) FROM t",
    # subscripts
    "SELECT a[1",
    "SELECT a[",
    "SELECT a[1 2]",
    "SELECT a[1] 2",
    # casts
    "SELECT CAST(x AS array<int)",
    "SELECT CAST(x AS map<int>)",
    "SELECT CAST(x AS struct<a int,>)",
    "SELECT CAST(x AS decimal(10,2)",
    "SELECT CAST(x AS decimal(10,2) y)",
    "SELECT CAST(x AS array<int>>)",
    "SELECT CAST(x AS array<int>",
    "SELECT CAST(x AS foo x)",
    "SELECT CAST(x AS int x)",
    "SELECT CAST(x AS int",
    "SELECT CAST(f(a) AS int",
    "SELECT CAST(x AS int) 2",
    # hints
    "SELECT /*+ x(a b) */ * FROM t",
    "SELECT /*+ x(a, b c) */ * FROM t",
    "SELECT /*+ x(a */ * FROM t",
    # clause keywords at the end of the input
    "SELECT a FROM t ORDER",
    "SELECT a FROM t ORDER BY",
    "SELECT a FROM t GROUP BY",
    "SELECT a FROM t SORT BY",
    "SELECT a FROM t DISTRIBUTE BY",
    "SELECT a FROM t CLUSTER BY",
    "SELECT a FROM t HAVING",
    "SELECT a FROM t WHERE",
    "SELECT a FROM t LIMIT",
    "SELECT a FROM t WINDOW",
    "SELECT a FROM t LATERAL VIEW",
    "SELECT a FROM t UNION ALL",
    "SELECT a FROM t ORDER BY a,",
    "SELECT sum(a) OVER (PARTITION BY",
    "SELECT a FROM t ORDER BY a +",
    "SELECT a FROM t ORDER BY a DESC,",
    "SELECT a FROM t SORT BY a,",
    "SELECT a FROM t LIMIT 1 +",
    "SELECT a FROM t t2 ORDER BY a,",
    "SELECT a FROM t GROUP BY a,",
    "SELECT a FROM t WHERE a +",
    "SELECT * FROM (SELECT 1 ORDER BY a,",
    "SELECT x IN (SELECT 1 2)",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("remote")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    cases = []
    for sql in CASES:
        try:
            spark.sql(sql).collect()
            error = None
        except PySparkException as e:
            error = e
        if error is not None and error.getCondition() and error.getCondition().startswith("PARSE"):
            case = {
                "sql": sql,
                "parses": False,
                "error_class": error.getCondition(),
                "message_parameters": error.getMessageParameters(),
            }
            # Every case must show its position in the message (errors Spark
            # raises while building the tree carry none over Connect).
            m = re.search(r"\(line (\d+), pos (\d+)\)", str(error))
            if not m:
                raise SystemExit(f"no position for {sql!r}")
            case["line"], case["pos"] = int(m[1]), int(m[2])
        else:
            # Parsed: any later (analysis) error is not a syntax matter.
            case = {"sql": sql, "parses": True}
        cases.append(case)
    doc = {"format_version": 1, "spark_version": spark.version, "generator": "tests/scripts/gen_sql_parse_errors.py", "cases": cases}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
