#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the DISTRIBUTE BY / CLUSTER BY evaluation golden fixture.

Question answered: when does Spark evaluate the partition expressions of `RepartitionByExpression`
(so that a failing expression raises), and when does the optimizer or planner drop the shuffle?
Every expectation comes from a REAL Spark 4.2.0 run (Spark Connect), never from the source or memory.
Per case it records the sorted result rows or the error (condition, SQLSTATE, message), and for the
`explain` cases the optimized logical plan (`EXPLAIN EXTENDED`, "Optimized Logical Plan" section).

Usage (from repo root), against the shared Spark 4.2.0 Connect oracle:

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \
      python tests/scripts/gen_distribute_by_goldens.py

or on a machine with a JDK and no oracle, a local Spark (`--local`, PySpark 4.2.0).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_distribute_by_goldens.py"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")

# `x` is a string column; CAST(x AS INT) fails for the row 'x' under ANSI (a RUNTIME failure).
# `x + 'a'` is NOT a clean runtime probe: Cast('a' AS DOUBLE) is foldable and may fail while optimizing.
VIEWS = {
    "t": "SELECT * FROM VALUES ('1'), ('2'), ('x') AS t(x)",
    "u": "SELECT * FROM VALUES (1), (2) AS u(k)",
    "ok": "SELECT * FROM VALUES ('1'), ('2') AS ok(x)",
    "n": "SELECT * FROM VALUES (0, 'abc'), (1, 'def') AS n(x, y)",
}
NO_ANSI = {"spark.sql.ansi.enabled": "false"}
NO_AQE = {"spark.sql.adaptive.enabled": "false"}

RUN = [
    # 1. basic: the expression is evaluated when the shuffle runs
    "SELECT x FROM t DISTRIBUTE BY x + 'a'",
    "SELECT x FROM t CLUSTER BY x + 'a'",
    "SELECT x FROM t DISTRIBUTE BY x + 'a' SORT BY x",
    "SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)",
    "SELECT x FROM t CLUSTER BY CAST(x AS INT)",
    "SELECT x FROM t DISTRIBUTE BY CAST(x AS INT) SORT BY x",
    "SELECT x FROM ok DISTRIBUTE BY CAST(x AS INT)",
    # 2. pruned above
    "SELECT count(*) FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT))",
    "SELECT 1 FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT))",
    # 3. filter above removes the failing row
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) WHERE x <> 'x'",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) WHERE rand() < 2 AND x <> 'x'",
    # 4. stacked
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) DISTRIBUTE BY x",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY x) DISTRIBUTE BY CAST(x AS INT)",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) CLUSTER BY x",
    "SELECT * FROM (SELECT x FROM t ORDER BY x) DISTRIBUTE BY CAST(x AS INT)",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) ORDER BY x",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT) SORT BY x) DISTRIBUTE BY x",
    # 5. limit above
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) LIMIT 1",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) LIMIT 0",
    # 6. under aggregate / join
    "SELECT x, count(*) FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) GROUP BY x",
    "SELECT count(*) FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) a JOIN u ON a.x = u.k",
    "SELECT u.k FROM u JOIN (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) a ON a.x = u.k",
    "SELECT x, count(*) FROM t GROUP BY x DISTRIBUTE BY CAST(x AS INT)",
    # 7. literal / foldable / non-deterministic expressions
    "SELECT x FROM t DISTRIBUTE BY 1",
    "SELECT x FROM t DISTRIBUTE BY CAST('a' AS INT)",
    "SELECT x FROM t WHERE false DISTRIBUTE BY CAST('a' AS INT)",
    "SELECT x FROM t DISTRIBUTE BY rand()",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT) + rand()) WHERE x <> 'x'",
    "SELECT x FROM t DISTRIBUTE BY CASE WHEN x = 'x' THEN 0 ELSE CAST(x AS INT) END",
    "SELECT x FROM t DISTRIBUTE BY 1 / 0",
    # 8. empty and one-row input
    "SELECT x FROM t WHERE false DISTRIBUTE BY CAST(x AS INT)",
    "SELECT * FROM (SELECT x FROM t WHERE false DISTRIBUTE BY CAST(x AS INT)) a",
    "SELECT 'x' AS x DISTRIBUTE BY CAST(x AS INT)",
    "SELECT * FROM (SELECT 'x' AS x DISTRIBUTE BY CAST(x AS INT))",
    "SELECT * FROM VALUES ('x') AS v(x) DISTRIBUTE BY CAST(x AS INT)",
    # 9. same error as a projection (message + context)
    "SELECT CAST(x AS INT) FROM t",
    # 10. an integral column: `x + 'a'` casts 'a' to a number per row
    "SELECT x FROM n DISTRIBUTE BY x + 'a'",
    "SELECT count(*) FROM (SELECT x FROM n DISTRIBUTE BY x + 'a')",
    # 11. on a correlated subquery's path (Spark's Category 2, like Sort)
    "SELECT k FROM u WHERE EXISTS (SELECT 1 FROM n WHERE n.x = u.k DISTRIBUTE BY n.x)",
    "SELECT k FROM u WHERE k IN (SELECT x FROM n WHERE n.x = u.k DISTRIBUTE BY x)",
    "SELECT k, (SELECT max(x) FROM n WHERE n.x = u.k DISTRIBUTE BY 1) FROM u",
    "SELECT k FROM u WHERE EXISTS (SELECT 1 FROM t WHERE t.x = CAST(u.k AS STRING) DISTRIBUTE BY CAST(t.x AS INT))",
    "SELECT k, (SELECT max(x) FROM t WHERE t.x = CAST(u.k AS STRING) DISTRIBUTE BY CAST(t.x AS INT)) FROM u",
    "SELECT k FROM u WHERE EXISTS (SELECT 1 FROM n WHERE n.x = u.k DISTRIBUTE BY u.k)",
    # 12. a sort over a repartition over an aggregate
    "SELECT y, count(*) AS c FROM n GROUP BY y DISTRIBUTE BY y SORT BY c",
    "SELECT y, count(*) AS c FROM n GROUP BY y DISTRIBUTE BY y SORT BY max(x)",
    "SELECT count(*) AS c FROM n GROUP BY y DISTRIBUTE BY c SORT BY y",
    "SELECT y, count(*) AS c FROM n GROUP BY y HAVING count(*) > 0 DISTRIBUTE BY y SORT BY max(x)",
    "SELECT k, (SELECT count(*) FROM n WHERE n.x = u.k DISTRIBUTE BY 1) FROM u",
]
# Each RUN query is also run with ANSI off and with AQE off.
CONFS = [{}, NO_ANSI, NO_AQE]

EXPLAIN = [
    "SELECT count(*) FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT))",
    "SELECT 1 FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT))",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) WHERE x <> 'x'",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) DISTRIBUTE BY x",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY x) DISTRIBUTE BY CAST(x AS INT)",
    "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) LIMIT 1",
    "SELECT x FROM t DISTRIBUTE BY 1",
    "SELECT x FROM t WHERE false DISTRIBUTE BY CAST(x AS INT)",
    "SELECT x FROM t CLUSTER BY CAST(x AS INT)",
]


def error_of(e) -> dict:
    cond = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", None)
    out = {
        "class": cond() if cond else "UNCLASSIFIED",
        "sqlstate": e.getSqlState() if hasattr(e, "getSqlState") else None,
        "message": str(e),
    }
    ctx = e.getQueryContext() if hasattr(e, "getQueryContext") else []
    if ctx:
        q = ctx[0]
        out["context"] = {"fragment": q.fragment()}
        if hasattr(q, "startIndex"):
            out["context"].update(start=q.startIndex(), stop=q.stopIndex())
    return out


def wrapped_error(e) -> dict:
    """An error Spark raised inside another (a broadcast's `ExecutionException`), which local mode
    hands over as a bare `Py4JJavaError`: its class and SQLSTATE from the message, no context."""
    text = str(e)
    cond = re.search(r"\[([A-Z][A-Z0-9_.]+)\]", text)
    state = re.search(r"SQLSTATE: ([0-9A-Z]{5})", text)
    return {
        "class": cond.group(1) if cond else "UNCLASSIFIED",
        "sqlstate": state.group(1) if state else None,
        "message": text.splitlines()[1] if len(text.splitlines()) > 1 else text,
        "wrapped": True,
    }


def run(spark, sql: str) -> dict:
    from pyspark.errors import PySparkException

    try:
        rows = [list(map(lambda v: None if v is None else str(v), r)) for r in spark.sql(sql).collect()]
    except PySparkException as e:
        return {"error": error_of(e)}
    except Exception as e:
        # py4j's `Py4JJavaError` (local mode), matched by name: py4j is pyspark's dependency, not ours.
        if type(e).__name__ != "Py4JJavaError":
            raise
        return {"error": wrapped_error(e)}
    return {"rows": sorted(rows, key=lambda r: [("" if v is None else v) for v in r])}


def explain(spark, sql: str) -> dict:
    from pyspark.errors import PySparkException

    try:
        text = spark.sql("EXPLAIN EXTENDED " + sql).collect()[0][0]
    except PySparkException as e:
        return {"error": error_of(e)}
    m = re.search(r"== Optimized Logical Plan ==\n(.*?)\n\n== Physical Plan ==", text, re.S)
    return {"optimized_plan": (m.group(1) if m else text).splitlines()}


def main() -> None:
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://127.0.0.1:15012")
    ap.add_argument(
        "--local",
        action="store_true",
        help="local[2] Spark instead of Connect (benchmark server: ops/scripts/spark-local.sh)",
    )
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/distribute-by-goldens.json")
    args = ap.parse_args()
    if args.local:
        spark = (
            SparkSession.builder.master("local[2]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.catalogImplementation", "in-memory")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.remote(args.remote).getOrCreate()
    if spark.version != "4.2.0":
        raise SystemExit(f"expected Spark 4.2.0, found {spark.version}")
    for name, query in VIEWS.items():
        spark.sql(f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {query}")

    cases = []
    for conf in CONFS:
        for k, v in conf.items():
            spark.conf.set(k, v)
        try:
            for sql in RUN:
                cases.append({"sql": sql, "conf": conf, **run(spark, sql)})
            if not conf:
                for sql in EXPLAIN:
                    cases.append({"sql": sql, "conf": conf, "explain": True, **explain(spark, sql)})
        finally:
            for k in conf:
                spark.conf.unset(k)
    for name in VIEWS:
        spark.sql(f"DROP VIEW IF EXISTS {name}")

    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": spark.version,
        "generator": GENERATOR,
        "views": VIEWS,
        "cases": cases,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
