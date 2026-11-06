#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate tests/fixtures/spark-4.2.0/pivot-empty-cells.json (kP2).

Every expectation is produced by a REAL Spark 4.2.0 run (local PySpark, in-memory catalog, UTC), never
transcribed. Records per case: rows (strings, sorted), output schema (name, type, nullable), the error class
if any, and `path` ("pivot_first" / "if_fallback", read from the analyzed plan: PivotFirst present or not).

Usage (from repo root; PySpark 4.2.0 + JDK 21 on PATH, JAVA_HOME and SPARK_LOCAL_IP=127.0.0.1 set):
    flock "$KRISHIV_DATA/.bench.lock" python tests/scripts/gen_pivot_empty_cells.py \
        --out tests/fixtures/spark-4.2.0/pivot-empty-cells.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWS = {
    # t(a,b): the bug report table.
    "t": "SELECT * FROM VALUES (1,2),(1,3),(1,4) AS t(a,b)",
    # g: group g, pivot column p (NULL rows), value v (NULL rows), string s.
    "g": "SELECT * FROM VALUES (1,10,5,'x'),(1,20,NULL,'y'),(1,NULL,7,'z'),(2,10,3,'w'),(2,NULL,NULL,'q'),(3,30,1,'r')"
    " AS g(g,p,v,s)",
    "f": "SELECT * FROM VALUES (1,CAST(0.0 AS DOUBLE),1),(1,CAST('NaN' AS DOUBLE),2),(1,CAST(-0.0 AS DOUBLE),3),"
    "(2,CAST(1.5 AS DOUBLE),4),(2,CAST(NULL AS DOUBLE),5) AS f(g,p,v)",
    "s": "SELECT * FROM VALUES (1,'a',1),(1,'b',2),(1,NULL,3),(2,'a',4),(2,'B',5) AS s(g,p,v)",
    "d": "SELECT * FROM VALUES (1,10,CAST(1.25 AS DECIMAL(10,2)),DATE'2020-01-01'),"
    "(1,20,CAST(2.50 AS DECIMAL(10,2)),DATE'2021-02-03'),"
    "(2,10,CAST(NULL AS DECIMAL(10,2)),CAST(NULL AS DATE)) AS d(g,p,x,dt)",
    "n": "SELECT * FROM VALUES (1,1,1.0),(1,1,1.0) AS n(g,p,v)",
    "tt": "SELECT * FROM VALUES (1,TIMESTAMP'2020-01-01 00:00:00',1),(2,TIMESTAMP'2020-01-01 00:00:00',2) AS tt(g,p,v)",
    "b": "SELECT * FROM VALUES (1,X'01',1),(1,X'02',2),(2,X'01',3) AS b(g,p,v)",
}

# (id, sql, conf, note)
CASES = [
    ("bug-ungrouped-count-star", "SELECT * FROM (SELECT a FROM t) PIVOT (count(*) FOR a IN (1, 2))", {}, "issue repro"),
    ("bug-grouped-count-star", "SELECT * FROM t PIVOT (count(*) FOR a IN (1, 2))", {}, "issue repro, group by b"),
    ("pf-count-x", "SELECT * FROM g PIVOT (count(v) FOR p IN (10, 20, 99))", {}, "count(col) PivotFirst"),
    ("pf-count-star", "SELECT * FROM (SELECT g, p FROM g) PIVOT (count(*) FOR p IN (10, 20, 99))", {}, ""),
    ("pf-count-if", "SELECT * FROM g PIVOT (count_if(v > 4) FOR p IN (10, 20, 99))", {}, ""),
    ("pf-sum", "SELECT * FROM g PIVOT (sum(v) FOR p IN (10, 20, 99))", {}, "sum of all-NULL cell is NULL both paths"),
    ("pf-approx-count-distinct", "SELECT * FROM g PIVOT (approx_count_distinct(v) FOR p IN (10, 20, 99))", {}, ""),
    (
        "pf-mix-count-sum",
        "SELECT * FROM g PIVOT (count(v) AS c, sum(v) AS s FOR p IN (10 AS x, 20 AS y, 99 AS z))",
        {},
        "multiple aggregates with aliases",
    ),
    (
        "pf-mix-count-no-alias",
        "SELECT * FROM g PIVOT (count(v), sum(v) FOR p IN (10, 20))",
        {},
        "names without aliases",
    ),
    (
        "pf-null-pivot-value",
        "SELECT * FROM g PIVOT (count(v) FOR p IN (10, NULL))",
        {},
        "IN (NULL): matches NULL pivot column rows?",
    ),
    ("pf-null-pivot-value-sum", "SELECT * FROM g PIVOT (sum(v) FOR p IN (NULL, 10))", {}, ""),
    (
        "pf-cast-pivot-value",
        "SELECT * FROM g PIVOT (count(v) FOR p IN ('10', 20.0, 99))",
        {},
        "values cast to the column type",
    ),
    (
        "pf-empty-input",
        "SELECT * FROM (SELECT * FROM g WHERE g > 100) PIVOT (count(v) FOR p IN (10, 20))",
        {},
        "no rows at all",
    ),
    ("pf-ungrouped-single-group", "SELECT * FROM (SELECT p, v FROM g) PIVOT (count(v) FOR p IN (10, 99))", {}, ""),
    ("pf-decimal-sum", "SELECT * FROM d PIVOT (sum(x) FOR p IN (10, 20, 99))", {}, "decimal is PivotFirst-eligible"),
    ("pf-decimal-count", "SELECT * FROM d PIVOT (count(x) FOR p IN (10, 20, 99))", {}, ""),
    ("pf-bool", "SELECT * FROM g PIVOT (bool_or(v > 4) FOR p IN (10, 20, 99))", {}, "boolean aggregate"),
    (
        "pf-float-pivot-col",
        "SELECT * FROM f PIVOT (sum(v) FOR p IN (0.0D, CAST('NaN' AS DOUBLE), 1.5D))",
        {},
        "PivotFirst, float pivot column: 0.0 value matches the -0.0 row?",
    ),
    ("pf-float-negzero-value", "SELECT * FROM f PIVOT (sum(v) FOR p IN (-0.0D, 1.5D))", {}, "-0.0 pivot value"),
    (
        "pf-float-nan-value",
        "SELECT * FROM f PIVOT (sum(v) FOR p IN (CAST('NaN' AS DOUBLE), 1.5D))",
        {},
        "NaN pivot value",
    ),
    (
        "pf-float-both-zeros",
        "SELECT * FROM f PIVOT (sum(v) FOR p IN (0.0D, -0.0D, 1.5D))",
        {},
        "both zeros as pivot values (collapse into one HashMap key?)",
    ),
    ("pf-float-dup-values", "SELECT * FROM f PIVOT (sum(v) FOR p IN (1.5D, 1.5D))", {}, "duplicate pivot values"),
    (
        "fb-float-dup-values",
        "SELECT * FROM f PIVOT (sum(v) AS s, collect_list(v) AS l FOR p IN (1.5D, 1.5D))",
        {},
        "duplicate pivot values on fallback",
    ),
    ("pf-int-dup-values", "SELECT * FROM g PIVOT (sum(v) FOR p IN (10, 10))", {}, "duplicate pivot values, int"),
    (
        "fb-float-pivot-col",
        "SELECT * FROM f PIVOT (sum(v) AS s, collect_list(v) AS l FOR p IN (0.0D, -0.0D, CAST('NaN' AS DOUBLE), 1.5D))",
        {},
        "same on the fallback path (<=>)",
    ),
    (
        "pf-string-pivot-col",
        "SELECT * FROM s PIVOT (count(v) FOR p IN ('a', 'b', 'A'))",
        {},
        "string pivot col, default collation",
    ),
    ("pf-string-pivot-col-null", "SELECT * FROM s PIVOT (count(v) FOR p IN ('a', NULL))", {}, ""),
    (
        "fb-string-max",
        "SELECT * FROM g PIVOT (count(v) AS c, max(s) AS m FOR p IN (10, 20, 99))",
        {},
        "string-typed aggregate forces ifExpr path for ALL aggregates; count becomes 0",
    ),
    (
        "fb-collect-list-alone",
        "SELECT * FROM g PIVOT (collect_list(v) FOR p IN (10, 20, 99))",
        {},
        "array aggregate: fallback",
    ),
    (
        "fb-count-with-collect-list",
        "SELECT * FROM g PIVOT (count(v) AS c, collect_list(v) AS l FOR p IN (10, 20, 99))",
        {},
        "",
    ),
    (
        "fb-count-star-with-collect-list",
        "SELECT * FROM g PIVOT (count(*) AS c, collect_list(v) AS l FOR p IN (10, 20, 99))",
        {},
        "",
    ),
    (
        "fb-null-pivot-value",
        "SELECT * FROM g PIVOT (count(v) AS c, max(s) AS m FOR p IN (10, NULL))",
        {},
        "NULL pivot value on fallback",
    ),
    ("fb-max-string-alone", "SELECT * FROM g PIVOT (max(s) FOR p IN (10, 20, 99))", {}, ""),
    ("fb-date-max", "SELECT * FROM d PIVOT (max(dt) FOR p IN (10, 20, 99))", {}, "date is NOT PivotFirst-eligible"),
    ("fb-date-count", "SELECT * FROM d PIVOT (count(dt) AS c, max(dt) AS m FOR p IN (10, 20, 99))", {}, ""),
    (
        "fb-timestamp-pivot-col-count",
        "SELECT * FROM tt PIVOT (count(v) AS c, first(v) AS f FOR p IN "
        "(TIMESTAMP'2020-01-01 00:00:00', TIMESTAMP'2030-01-01 00:00:00'))",
        {},
        "first() is int: eligible; kept as control",
    ),
    (
        "fb-interval-max",
        "SELECT * FROM (SELECT g, p, make_interval(0,0,0,v) AS iv FROM g) PIVOT (max(iv) FOR p IN (10, 99))",
        {},
        "interval not eligible",
    ),
    (
        "fb-first",
        "SELECT * FROM g PIVOT (first(s) FOR p IN (10, 20, 99))",
        {},
        "First/Last special-cased with ignoreNulls",
    ),
    ("pf-first-int", "SELECT * FROM g PIVOT (first(v) FOR p IN (10, 20, 99))", {}, "first on int is PivotFirst"),
    (
        "pf-nonmatching-arg-error",
        "SELECT * FROM g PIVOT (sum(1 / (v - 5)) FOR p IN (10, 20))",
        {"spark.sql.ansi.enabled": "true"},
        "arg error on rows of another pivot value: PivotFirst evaluates all rows in stage 1",
    ),
    (
        "fb-nonmatching-arg-error",
        "SELECT * FROM g PIVOT (sum(1 / (v - 5)) AS s, collect_list(v) AS l FOR p IN (10, 20))",
        {"spark.sql.ansi.enabled": "true"},
        "same arg on the fallback path: error only on matching rows (ifExpr)",
    ),
    (
        "pf-ansi-off-div",
        "SELECT * FROM g PIVOT (sum(1 / (v - 5)) FOR p IN (10, 20))",
        {"spark.sql.ansi.enabled": "false"},
        "",
    ),
    (
        "err-unorderable-pivot-col",
        "SELECT * FROM (SELECT g, map(1,2) AS m, v FROM g) PIVOT (sum(v) FOR m IN (1))",
        {},
        "error class",
    ),
    (
        "pf-array-pivot-col",
        "SELECT * FROM (SELECT g, array(p) AS ap, v FROM g) PIVOT (count(v) FOR ap IN (array(10), array(99)))",
        {},
        "TreeMap path (non-atomic pivot column), null elements",
    ),
    (
        "pf-byte-short-long",
        "SELECT * FROM (SELECT g, p, CAST(v AS TINYINT) AS bv FROM g) PIVOT (sum(bv) FOR p IN (10, 99))",
        {},
        "sum->bigint",
    ),
    ("pf-max-int-null-cell", "SELECT * FROM g PIVOT (max(v) FOR p IN (10, 20, 99))", {}, ""),
    ("pf-avg", "SELECT * FROM g PIVOT (avg(v) FOR p IN (10, 20, 99))", {}, "double"),
    # Review round 1: duplicates of values the analyzer cannot fold, NaN twice, ANSI off, empty input, cast errors,
    # binary pivot columns.
    (
        "pf-empty-input-ungrouped",
        "SELECT * FROM (SELECT p, v FROM g WHERE p > 100) PIVOT (count(v) FOR p IN (10))",
        {},
        "ungrouped PivotFirst over empty input: the stage-2 global aggregate",
    ),
    (
        "pf-ts-dup-values",
        "SELECT * FROM tt PIVOT (count(v) FOR p IN (TIMESTAMP'2020-01-01 00:00:00', TIMESTAMP'2020-01-01 00:00:00'))",
        {},
        "duplicate timestamp values",
    ),
    (
        "pf-ts-dup-cast-value",
        "SELECT * FROM tt PIVOT (count(v) FOR p IN (TIMESTAMP'2020-01-01 00:00:00', '2020-01-01 00:00:00'))",
        {},
        "a string value equal to a timestamp value after the cast",
    ),
    (
        "pf-ts-distinct-values",
        "SELECT * FROM tt PIVOT (count(v) FOR p IN (TIMESTAMP'2020-01-01 00:00:00', '2021-01-01 00:00:00'))",
        {},
        "distinct timestamp values, one a string",
    ),
    (
        "pf-float-nan-twice",
        "SELECT * FROM f PIVOT (sum(v) FOR p IN (CAST('NaN' AS DOUBLE), CAST('NaN' AS DOUBLE)))",
        {},
        "two NaN values: separate keys?",
    ),
    (
        "pf-int-dup-ansi-off",
        "SELECT * FROM g PIVOT (sum(v) FOR p IN (10, 20, 10))",
        {"spark.sql.ansi.enabled": "false"},
        "duplicate values with ANSI off",
    ),
    (
        "pf-int-dup-empty-input",
        "SELECT * FROM (SELECT * FROM g WHERE g > 100) PIVOT (sum(v) FOR p IN (10, 10))",
        {},
        "duplicate values, no row produced",
    ),
    (
        "pf-cast-error-value",
        "SELECT * FROM (SELECT * FROM g WHERE g > 100) PIVOT (sum(v) FOR p IN ('abc'))",
        {},
        "a value whose cast to the pivot type fails, empty input",
    ),
    (
        "fb-cast-error-value",
        "SELECT * FROM (SELECT * FROM g WHERE g > 100) PIVOT (max(s) FOR p IN ('abc'))",
        {},
        "the If fallback also evaluates the values at analysis",
    ),
    (
        "pf-binary-pivot-col",
        "SELECT * FROM b PIVOT (sum(v) FOR p IN (X'01', X'03'))",
        {},
        "binary pivot column (4.2: TreeMap, matched by content)",
    ),
    (
        "pf-binary-dup-values",
        "SELECT * FROM b PIVOT (sum(v) FOR p IN (X'01', X'01'))",
        {},
        "duplicate binary values",
    ),
]


def err_class(e: Exception) -> dict:
    c = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", None)
    return {
        "error_class": c() if c else type(e).__name__,
        "sql_state": state() if callable(state := getattr(e, "getSqlState", None)) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("pivot-empty-cells")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    assert spark.version == "4.2.0", spark.version
    for name, sql in VIEWS.items():
        spark.sql(sql).createOrReplaceTempView(name)
    out = []
    for cid, sql, conf, note in CASES:
        for k, v in conf.items():
            spark.conf.set(k, v)
        rec = {"id": cid, "sql": sql, "conf": conf, "note": note}
        try:
            df = spark.sql(sql)
            analyzed = df._jdf.queryExecution().analyzed().toString()
            rec["path"] = "pivot_first" if "pivotfirst" in analyzed.lower() else "if_fallback"
            rec["output"] = [
                {"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in df.schema.fields
            ]
            rows = df.collect()
            rec["ok"] = True
            rec["rows"] = sorted("|".join("NULL" if v is None else str(v) for v in r) for r in rows)
        except Exception as e:
            rec["ok"] = False
            rec.update(err_class(e))
        out.append(rec)
        for k in conf:
            spark.conf.unset(k)
    spark.stop()
    doc = {
        "format_version": 1,
        "spark_version": "4.2.0",
        "generator": "tests/scripts/gen_pivot_empty_cells.py",
        "notes": "PIVOT cells with no matching row: PivotFirst path (all aggregates byte/short/int/long/float/double/"
        "boolean/decimal) vs the If(pivot <=> value) fallback. Rows are `|`-joined strings, sorted; "
        "`views` are temp views.",
        "views": VIEWS,
        "cases": out,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
