#!/usr/bin/env python3
"""Generate the analyzer golden fixture (P2-T071a).

Every expectation comes from a REAL Spark 4.2 run (the `krishiv-bench` container's Spark Connect server),
never from docs or memory. The script creates empty tables (the TPC-H schema and a few small tables) in
`spark_catalog.default`, then analyzes each case with `DataFrame.schema` (Spark Connect `AnalyzePlan`, so
the server runs its full analyzer) and records either the output columns (name, `simpleString` type,
nullability) or the error condition, SQLSTATE, message parameters and query context (start, stop,
fragment) Spark raised. Cases are the 22 TPC-H queries, a name-resolution corpus (SQL, with per-case
configs) and DataFrame `plan_id` cases (column references such as `df["a"]`, built through the
Connect client). The table schemas Spark reports are recorded too, so tests resolve against the same
catalog. Output is deterministic and contains no local paths or hostnames.

Usage (from repo root; take the bench lock, start Spark, stop it afterwards; see bench/README.md):

    docker compose -f bench/baselines/docker-compose.yml up -d spark
    flock "$KRISHIV_DATA/.bench.lock" nice -n 19 \
      uv run --project bench python tests/scripts/gen_analyzer_goldens.py \
        --out tests/fixtures/spark-4.2.0/analyzer-goldens.json
    docker compose -f bench/baselines/docker-compose.yml stop spark
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_analyzer_goldens.py"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")

MONEY = "DECIMAL(15,2)"
TABLES = {
    "part": "p_partkey BIGINT, p_name STRING, p_mfgr STRING, p_brand STRING, p_type STRING, p_size INT, "
    f"p_container STRING, p_retailprice {MONEY}, p_comment STRING",
    "supplier": "s_suppkey BIGINT, s_name STRING, s_address STRING, s_nationkey BIGINT, s_phone STRING, "
    f"s_acctbal {MONEY}, s_comment STRING",
    "partsupp": f"ps_partkey BIGINT, ps_suppkey BIGINT, ps_availqty INT, ps_supplycost {MONEY}, ps_comment STRING",
    "customer": "c_custkey BIGINT, c_name STRING, c_address STRING, c_nationkey BIGINT, c_phone STRING, "
    f"c_acctbal {MONEY}, c_mktsegment STRING, c_comment STRING",
    "orders": f"o_orderkey BIGINT, o_custkey BIGINT, o_orderstatus STRING, o_totalprice {MONEY}, o_orderdate DATE, "
    "o_orderpriority STRING, o_clerk STRING, o_shippriority INT, o_comment STRING",
    "lineitem": "l_orderkey BIGINT, l_partkey BIGINT, l_suppkey BIGINT, l_linenumber INT, "
    f"l_quantity {MONEY}, l_extendedprice {MONEY}, l_discount {MONEY}, l_tax {MONEY}, l_returnflag STRING, "
    "l_linestatus STRING, l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct STRING, "
    "l_shipmode STRING, l_comment STRING",
    "nation": "n_nationkey BIGINT, n_name STRING, n_regionkey BIGINT, n_comment STRING",
    "region": "r_regionkey BIGINT, r_name STRING, r_comment STRING",
    "t": "a INT, b STRING, c BIGINT",
    "u": "a INT, d STRING",
    "wide": "col1 INT, col2 INT, col3 INT, col4 INT, col5 INT, col6 INT, col7 INT, colx INT",
    "MixedCase": "Id INT, Name STRING",
}

CASE_SENSITIVE = {"spark.sql.caseSensitive": "true"}
NO_ALIASES = {"spark.sql.groupByAliases": "false"}
NO_ORDINALS = {"spark.sql.groupByOrdinal": "false", "spark.sql.orderByOrdinal": "false"}
NO_LCA = {"spark.sql.lateralColumnAlias.enableImplicitResolution": "false"}

CORPUS = [
    # unresolved and ambiguous columns, suggestions and their order
    ("SELECT x FROM t", {}),
    ("SELECT colz FROM wide", {}),
    ("SELECT t.x FROM t", {}),
    ("SELECT v.a FROM t", {}),
    ("SELECT a FROM t JOIN u ON t.a = u.a", {}),
    ("SELECT a FROM t, u", {}),
    ("SELECT t.a, u.a FROM t JOIN u ON t.a = u.a", {}),
    ("SELECT default.t.a, spark_catalog.default.t.b FROM t", {}),
    ("SELECT a.b FROM t", {}),
    ("SELECT t.a.b FROM t", {}),
    ("SELECT x FROM t WHERE a = 1", {}),
    ("SELECT a FROM t WHERE x = 1", {}),
    ("SELECT a FROM t ORDER BY x", {}),
    # case sensitivity
    ("SELECT A, T.B FROM t", {}),
    ("SELECT A FROM t", CASE_SENSITIVE),
    ("SELECT id, NAME FROM mixedcase", {}),
    ("SELECT id FROM MixedCase", CASE_SENSITIVE),
    ("SELECT Id FROM MixedCase", CASE_SENSITIVE),
    ("WITH Q AS (SELECT 1 AS z FROM u) SELECT z FROM q", {}),
    # unknown functions and tables
    ("SELECT nosuch(a) FROM t", {}),
    ("SELECT nosuch(x) FROM missing_table", {}),
    ("SELECT * FROM missing_table", {}),
    ("SELECT * FROM default.missing_table", {}),
    ("SELECT * FROM t JOIN missing_table ON true", {}),
    # CTEs, derived tables, aliases
    ("WITH t AS (SELECT 1 AS z FROM u) SELECT z FROM t", {}),
    ("WITH t AS (SELECT 1 AS z FROM u) SELECT a FROM t", {}),
    ("WITH x AS (SELECT 1 AS p FROM u) SELECT * FROM (WITH x AS (SELECT 2 AS q FROM u) SELECT * FROM x)", {}),
    ("WITH x AS (SELECT a FROM t) SELECT * FROM x a JOIN x b ON a.a = b.a", {}),
    ("WITH x AS (SELECT a FROM t), y AS (SELECT a + 1 AS b FROM x) SELECT b FROM y JOIN x ON y.b = x.a", {}),
    ("SELECT s.a FROM (SELECT a FROM t) s", {}),
    ("SELECT t.a FROM (SELECT a FROM t) s", {}),
    ("SELECT a FROM (SELECT a FROM t)", {}),
    ("SELECT p, q FROM t AS x (p, q, r)", {}),
    ("SELECT * FROM t AS x (p, q)", {}),
    ("SELECT x.a FROM t x JOIN t y ON x.a = y.a", {}),
    # stars
    ("SELECT * FROM t", {}),
    ("SELECT t.* FROM t JOIN u ON t.a = u.a", {}),
    ("SELECT u.*, t.a FROM t JOIN u ON t.a = u.a", {}),
    ("SELECT x.* FROM t", {}),
    ("SELECT count(*) FROM t", {}),
    # USING joins
    ("SELECT * FROM t JOIN u USING (a)", {}),
    ("SELECT * FROM t LEFT JOIN u USING (a)", {}),
    ("SELECT * FROM t RIGHT JOIN u USING (a)", {}),
    ("SELECT * FROM t FULL JOIN u USING (a)", {}),
    ("SELECT * FROM t JOIN u USING (z)", {}),
    ("SELECT * FROM t JOIN u USING (b)", {}),
    # aggregates
    ("SELECT a + 1 AS k, count(*) FROM t GROUP BY k", {}),
    ("SELECT a + 1 AS k, count(*) FROM t GROUP BY k", NO_ALIASES),
    ("SELECT a, b FROM t GROUP BY a", {}),
    ("SELECT a + 1, sum(c) FROM t GROUP BY a + 1", {}),
    ("SELECT a FROM t GROUP BY a HAVING max(c) > 1", {}),
    ("SELECT a, max(c) AS m FROM t GROUP BY a HAVING m > 1", {}),
    ("SELECT a FROM t GROUP BY a HAVING b = 'x'", {}),
    ("SELECT sum(max(a)) FROM t", {}),
    ("SELECT a FROM t WHERE sum(a) > 1", {}),
    ("SELECT sum(a), b FROM t", {}),
    ("SELECT count(DISTINCT a), max(b) FROM t", {}),
    # ordinals, ORDER BY aliases and missing attributes
    ("SELECT a, count(*) FROM t GROUP BY 1", {}),
    ("SELECT a, count(*) FROM t GROUP BY 3", {}),
    ("SELECT a, count(*) FROM t GROUP BY 2", {}),
    ("SELECT *, 1 FROM t GROUP BY 1", {}),
    ("SELECT a, count(*) FROM t GROUP BY 1", NO_ORDINALS),
    ("SELECT a, b FROM t ORDER BY 2", {}),
    ("SELECT a, b FROM t ORDER BY 3", {}),
    ("SELECT a AS k FROM t ORDER BY k", {}),
    ("SELECT a FROM t ORDER BY b", {}),
    ("SELECT a, count(*) FROM t GROUP BY a ORDER BY count(*) DESC", {}),
    ("SELECT a FROM t GROUP BY a ORDER BY max(c)", {}),
    # lateral column aliases
    ("SELECT a + 1 AS k, k + 1 AS m FROM t", {}),
    ("SELECT a + 1 AS k, k + 1 AS m FROM t", NO_LCA),
    ("SELECT 1 AS k, 2 AS k, k + 1 FROM t", {}),
    ("SELECT a AS b, b FROM t", {}),
    ("SELECT max(a) AS m, m + 1 FROM t", {}),
    ("SELECT a + 1 AS k, k + 1 FROM t", {}),
    ("SELECT a + 1 AS k, k + 1 AS m, m * 2 AS n FROM t", {}),
    ("SELECT a, max(c) AS m, m + a FROM t GROUP BY a", {}),
    # subqueries and outer references
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a)", {}),
    ("SELECT a FROM t WHERE a IN (SELECT a FROM u WHERE d = b)", {}),
    ("SELECT (SELECT max(a) FROM u) FROM t", {}),
    ("SELECT (SELECT a, d FROM u) FROM t", {}),
    ("SELECT a FROM t WHERE (SELECT x FROM u) = 1", {}),
    # generated output names
    (
        "SELECT a + 1, -a, a = 1, NOT (a = 1), a <> 1, CAST(a AS STRING), 'x', 1.5, 2, 10L, a IS NULL, "
        "a IS NOT NULL, a IN (1, 2), a BETWEEN 1 AND 2, CASE WHEN a = 1 THEN 'y' ELSE 'z' END, "
        "substring(b, 1, 2), a * 2 - c / 3, b LIKE 'x%', a % 2, TRUE, DATE '2020-01-02', "
        "a <=> 1, a >= 1 AND a < 2 OR c > 3, extract(year FROM DATE '2020-01-02') FROM t",
        {},
    ),
    ("SELECT count(*), count(a), sum(a), avg(c), min(b), max(b), count(DISTINCT a) FROM t", {}),
    # WITH columns, views and others the analyzer rejects or handles
    ("SELECT * FROM t WHERE a = 1 AND b = 'x' OR c = 3", {}),
    ("SELECT a FROM t LIMIT 10", {}),
    # inner before outer, case-distinct CTEs, 4-part names, HAVING aliases
    ("SELECT a FROM t WHERE EXISTS (SELECT a FROM u WHERE a = 1)", {}),
    ("SELECT b FROM t WHERE EXISTS (SELECT 1 FROM u WHERE b = 'x')", {}),
    ("WITH T AS (SELECT 1 AS z FROM u) SELECT * FROM t", CASE_SENSITIVE),
    ("WITH T AS (SELECT 1 AS z FROM u) SELECT * FROM t", {}),
    ("SELECT spark_catalog.default.t.a FROM t", {}),
    ("SELECT spark_catalog.default.t.* FROM t", {}),
    ("SELECT a AS b, max(c) FROM t GROUP BY a HAVING b > 0", {}),
    ("SELECT a FROM t WHERE b LIKE 'x%' AND c IS NOT NULL ORDER BY a DESC NULLS FIRST", {}),
    ("SELECT t.a, d FROM t LEFT JOIN u ON t.a = u.a", {}),
    ("SELECT u.a, b FROM t RIGHT JOIN u ON t.a = u.a", {}),
    ("SELECT u.a, t.b FROM t FULL JOIN u ON t.a = u.a", {}),
    ("SELECT a FROM t LEFT SEMI JOIN u ON t.a = u.a", {}),
    ("SELECT d FROM t LEFT ANTI JOIN u ON t.a = u.a", {}),
    ("SELECT * FROM t CROSS JOIN u", {}),
    ("SELECT CAST(a AS BIGINT) + 1 AS x, TRY_CAST(b AS INT) FROM t", {}),
    ("SELECT count(a) FILTER (WHERE b = 'x') FROM t", {}),
    # lateral column aliases in projections and aggregates
    ("SELECT a + 1, `(a + 1)` + 1 FROM t", {}),
    ("SELECT a + 1 AS a, a + 1 FROM t", {}),
    ("SELECT a AS k, b AS k, k FROM t", {}),
    ("SELECT a AS k, k AS j, j FROM t", {}),
    ("SELECT a AS k, (SELECT max(k) FROM u) FROM t", {}),
    ("SELECT a AS k, k FROM t WHERE k > 1", {}),
    ("SELECT (SELECT 1 FROM u) AS k, k + 1 FROM t", {}),
    ("SELECT (SELECT max(a) FROM u WHERE u.a = t.a) AS s, s + 1 FROM t", {}),
    ("SELECT max(a) AS m, m FROM t", {}),
    ("SELECT count(*) AS n, n * 2 FROM t", {}),
    ("SELECT max(a) AS m, m + 1 AS n, n + 1 FROM t", {}),
    ("SELECT max(a) AS m, m + 1 FROM t HAVING m > 1", {}),
    ("SELECT a AS k, sum(k) FROM t", {}),
    ("SELECT max(a) AS m, sum(m) FROM t", {}),
    ("SELECT a, a AS k, max(k) FROM t GROUP BY a", {}),
    ("SELECT a AS k, sum(c) FILTER (WHERE k > 1) FROM t GROUP BY a", {}),
    ("SELECT sum(c) AS s, s + max(a) FROM t", {}),
    ("SELECT sum(c) AS s, s + max(a) AS z FROM t GROUP BY b", {}),
    ("SELECT a, max(c) AS m, m + c FROM t GROUP BY a", {}),
    ("SELECT sum(c) AS s, s + a FROM t GROUP BY a + 1", {}),
    ("SELECT a + 1 AS k, sum(c) AS s, k * s FROM t GROUP BY a + 1", {}),
    ("SELECT a + 1 AS k, k + 1 FROM t GROUP BY a", {}),
    ("SELECT a AS k, k + 1 FROM t GROUP BY a, k", {}),
    ("SELECT sum(c) AS s, s + 1 AS z FROM t GROUP BY a HAVING z > 2", {}),
    ("SELECT sum(c) AS s, s + 1 AS z FROM t GROUP BY a HAVING max(b) > 'x'", {}),
    ("SELECT sum(c) AS s, s + 1 AS z FROM t GROUP BY a ORDER BY max(b)", {}),
    ("SELECT sum(c) AS s, s + 1 FROM t GROUP BY a ORDER BY a", {}),
    ("SELECT max(a) AS m, m + c FROM t", {}),
    ("SELECT sum(c) AS s, s + a FROM t GROUP BY a", {}),
    ("SELECT sum(c) AS s, s + (a + 1) FROM t GROUP BY a + 1", {}),
    ("SELECT sum(c) AS s, s + max(a) AS z FROM t GROUP BY b ORDER BY max(a)", {}),
    ("SELECT sum(c) AS s, s + a AS z FROM t GROUP BY a ORDER BY a", {}),
    ("SELECT a AS k, CAST(k AS BIGINT) FROM t", {}),
    # GROUP BY / ORDER BY ordinals and their errors, in Spark's order
    ("SELECT a + 1 AS k, count(*) FROM t GROUP BY 1", {}),
    ("SELECT 1 AS k, count(*) FROM t GROUP BY 1", {}),
    ("SELECT b AS k, count(*) FROM t GROUP BY 1 HAVING k = 'x'", {}),
    ("SELECT a, count(*) FROM t GROUP BY 1, 2", {}),
    ("SELECT max(c) AS m FROM t GROUP BY 1", {}),
    ("SELECT max(c) AS m, m FROM t GROUP BY 2", {}),
    ("SELECT a AS k, k, count(*) FROM t GROUP BY 1, 2", {}),
    ("SELECT x, count(*) FROM t GROUP BY 3", {}),
    ("SELECT a, count(*) FROM t GROUP BY y, 3", {}),
    ("SELECT x FROM t GROUP BY y", {}),
    ("SELECT *, 1 FROM t GROUP BY 5", {}),
    ("SELECT * FROM t GROUP BY a", {}),
    ("SELECT a, b FROM t ORDER BY -1", {}),
    ("SELECT a, b FROM t ORDER BY 0", {}),
    ("SELECT a, b FROM t ORDER BY 1L", {}),
    ("SELECT a FROM t ORDER BY x, 2", {}),
    ("SELECT * FROM t ORDER BY 3", {}),
    ("SELECT a, count(*) FROM t GROUP BY a ORDER BY 2", {}),
    ("SELECT a, count(*) FROM t GROUP BY a HAVING count(*) > 1 ORDER BY 2", {}),
    ("SELECT a FROM t GROUP BY 1 HAVING count(*) > 1 ORDER BY 1", {}),
    ("SELECT a AS k, k + 1 AS j FROM t ORDER BY 2", {}),
    # ORDER BY columns outside the select list
    ("SELECT a FROM t WHERE c > 1 ORDER BY b, a + c", {}),
    ("SELECT a AS k FROM t ORDER BY b + k", {}),
    ("SELECT a FROM (SELECT a, b FROM t) s ORDER BY b", {}),
    ("SELECT a FROM t ORDER BY b LIMIT 1", {}),
    ("SELECT t.a FROM t JOIN u ON t.a = u.a ORDER BY d", {}),
    ("SELECT a AS k, k + 1 AS j FROM t ORDER BY c", {}),
    ("SELECT a AS k FROM t GROUP BY k ORDER BY k", {}),
    ("SELECT a AS b, c AS b FROM t ORDER BY b", {}),
    ("SELECT a AS k, b AS k FROM t ORDER BY k", {}),
    ("SELECT a FROM t GROUP BY a HAVING count(*) > 1 ORDER BY b", {}),
    ("SELECT max(c) FROM t GROUP BY a, b HAVING count(*) > 1 ORDER BY b", {}),
    ("SELECT max(c) AS m FROM t GROUP BY a HAVING count(*) > 1 ORDER BY a", {}),
]

TPCH_DIR = Path(__file__).resolve().parents[2] / "bench" / "queries" / "tpch"


def plan_id_cases(spark):
    """DataFrame cases: (id, description, thunk returning a DataFrame)."""
    from pyspark.sql import functions as F

    t = spark.table("t")
    u = spark.table("u")
    return [
        ("same_df", 't.select(t["a"])', lambda: t.select(t["a"])),
        ("other_df", 't.select(u["a"])', lambda: t.select(u["a"])),
        ("through_filter", 't.filter(t["a"] > 1).select(t["b"])', lambda: t.filter(t["a"] > 1).select(t["b"])),
        ("projected_away", 't.select("a").select(t["b"])', lambda: t.select("a").select(t["b"])),
        (
            "join_two",
            't.join(u, t["a"] == u["a"]).select(t["b"], u["d"])',
            lambda: t.join(u, t["a"] == u["a"]).select(t["b"], u["d"]),
        ),
        ("self_join", 't.join(t, "a").select(t["b"])', lambda: t.join(t, "a").select(t["b"])),
        ("self_join_cond", 't.join(t, t["a"] == t["a"])', lambda: t.join(t, t["a"] == t["a"])),
        ("star", 't.join(u, t["a"] == u["a"]).select(u["*"])', lambda: t.join(u, t["a"] == u["a"]).select(u["*"])),
        (
            "with_column",
            't.withColumn("a", t["a"] + 1).withColumn("e", F.lit(1))',
            lambda: t.withColumn("a", t["a"] + 1).withColumn("e", F.lit(1)),
        ),
        ("missing", 't.select(t["x"])', lambda: t.select(t["x"])),
        (
            "outer",
            't.where(u.where(u["a"] == t["a"].outer()).exists()).select(t["b"])',
            lambda: t.where(u.where(u["a"] == t["a"].outer()).exists()).select(t["b"]),
        ),
        # columns a projection dropped, found below it (missing references)
        (
            "missing_through_filter",
            't.select("a").where(F.col("a") > 1).orderBy("b")',
            lambda: t.select("a").where(F.col("a") > 1).orderBy("b"),
        ),
        ("missing_through_limit", 't.select("a").limit(1).orderBy("b")', lambda: t.select("a").limit(1).orderBy("b")),
        (
            "missing_two_sorts",
            't.select("a").orderBy("b").orderBy("c")',
            lambda: t.select("a").orderBy("b").orderBy("c"),
        ),
        (
            "missing_two_projects",
            't.select("a", "b").select("a").orderBy("c")',
            lambda: t.select("a", "b").select("a").orderBy("c"),
        ),
        (
            "missing_in_filter",
            't.select("a", "c").where(F.col("b") == "x")',
            lambda: t.select("a", "c").where(F.col("b") == "x"),
        ),
        (
            "missing_grouping",
            't.groupBy("a").agg(F.max("c").alias("m")).select("m").orderBy("a")',
            lambda: t.groupBy("a").agg(F.max("c").alias("m")).select("m").orderBy("a"),
        ),
        ("missing_after_alias", 't.select("a").alias("x").orderBy("b")', lambda: t.select("a").alias("x").orderBy("b")),
        (
            "lateral_alias",
            't.select((t["a"] + 1).alias("k"), (F.col("k") + 1).alias("m"))',
            lambda: t.select((t["a"] + 1).alias("k"), (F.col("k") + 1).alias("m")),
        ),
    ]


def record_error(e) -> dict:
    out = {"error_class": None, "sql_state": None, "message_parameters": {}}
    get = getattr
    cond = get(e, "getCondition", None) or get(e, "getErrorClass", None)
    out["error_class"] = cond() if cond else "UNCLASSIFIED"
    out["sql_state"] = e.getSqlState() if hasattr(e, "getSqlState") else None
    params = e.getMessageParameters() if hasattr(e, "getMessageParameters") else {}
    out["message_parameters"] = dict(sorted((params or {}).items()))
    ctx = e.getQueryContext() if hasattr(e, "getQueryContext") else []
    if ctx:
        q = ctx[0]
        kind = q.contextType().name if hasattr(q, "contextType") else ""
        if kind == "SQL":
            out["context"] = {"start": q.startIndex(), "stop": q.stopIndex(), "fragment": q.fragment()}
        else:
            out["context"] = {"fragment": q.fragment()}
    return out


def analyze(thunk) -> dict:
    from pyspark.errors import PySparkException

    try:
        schema = thunk().schema
    except PySparkException as e:
        return {"ok": False, **record_error(e)}
    return {
        "ok": True,
        "output": [{"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in schema.fields],
    }


def main() -> None:
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/analyzer-goldens.json")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    if spark.version != "4.2.0":
        raise SystemExit(f"expected Spark 4.2.0, found {spark.version}")
    spark.sql("CREATE DATABASE IF NOT EXISTS default")
    for name, cols in TABLES.items():
        spark.sql(f"DROP TABLE IF EXISTS {name}")
        spark.sql(f"CREATE TABLE {name} ({cols}) USING parquet")
    tables = {
        name: [
            {"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable}
            for f in spark.table(name).schema.fields
        ]
        for name in TABLES
    }

    def run_sql(sql: str, conf: dict) -> dict:
        for k, v in conf.items():
            spark.conf.set(k, v)
        try:
            return analyze(lambda: spark.sql(sql))
        finally:
            for k in conf:
                spark.conf.unset(k)

    tpch = []
    for path in sorted(TPCH_DIR.glob("q*.sql")):
        sql = path.read_text(encoding="utf-8").strip().rstrip(";")
        tpch.append({"id": path.stem, **run_sql(sql, {})})
    corpus = [{"sql": sql, "conf": conf, **run_sql(sql, conf)} for sql, conf in CORPUS]
    plan_ids = [{"id": i, "dataframe": d, **analyze(f)} for i, d, f in plan_id_cases(spark)]
    for name in TABLES:
        spark.sql(f"DROP TABLE IF EXISTS {name}")

    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": spark.version,
        "generator": GENERATOR,
        "notes": "Output columns or error (condition, SQLSTATE, message parameters, query context) of Spark's "
        "analyzer per case, over empty parquet tables in spark_catalog.default whose schemas are `tables`.",
        "tables": tables,
        "tpch": tpch,
        "corpus": corpus,
        "plan_ids": plan_ids,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(tpch)} TPC-H, {len(corpus)} corpus and {len(plan_ids)} plan_id cases -> {args.out}")


if __name__ == "__main__":
    main()
