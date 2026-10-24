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
    "g": "x INT, l BIGINT",
    "n": "ti TINYINT, si SMALLINT, i INT, l BIGINT, f FLOAT, d DOUBLE, dc DECIMAL(10,2), dw DECIMAL(38,10), "
    "s STRING, bo BOOLEAN, dt DATE, ts TIMESTAMP, tn TIMESTAMP_NTZ, bi BINARY",
}

CASE_SENSITIVE = {"spark.sql.caseSensitive": "true"}
NO_ALIASES = {"spark.sql.groupByAliases": "false"}
NO_ORDINALS = {"spark.sql.groupByOrdinal": "false", "spark.sql.orderByOrdinal": "false"}
NO_LCA = {"spark.sql.lateralColumnAlias.enableImplicitResolution": "false"}
NO_ANSI = {"spark.sql.ansi.enabled": "false"}
NO_LOSS = {"spark.sql.decimalOperations.allowPrecisionLoss": "false"}

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
    # a dropped column hides an outer one of its name
    ("SELECT * FROM (SELECT a, d AS b FROM u) o WHERE EXISTS (SELECT a FROM t ORDER BY b LIMIT 1)", {}),
    ("SELECT * FROM (SELECT a, d AS b FROM u) o WHERE EXISTS (SELECT a AS k, a AS k FROM t ORDER BY k LIMIT 1)", {}),
    ("SELECT max(c) FROM t GROUP BY a, b HAVING count(*) > 1 ORDER BY b", {}),
    ("SELECT max(c) AS m FROM t GROUP BY a HAVING count(*) > 1 ORDER BY a", {}),
    # type coercion (n has a column of each type): arithmetic, decimals and literals
    (
        "SELECT ti + si, i + l, f + f, f + i, d * l, dc + i, dc * dc, dw + dw, dc / i, i / l, i % ti, dc % i, "
        "1 + 1.5, dc + 1, dc * 2, 12345678901 + dc, -5 * dc, s + i, s * d, s / l, -s, -dc, NULL + i, NULL + NULL "
        "FROM n",
        {},
    ),
    ("SELECT dc * dw, dw / dc, dc + dw FROM n", NO_LOSS),
    ("SELECT dc + i, i / 2, dc * 2, -s FROM n", NO_ANSI),
    # comparisons, IN, BETWEEN, CASE, boolean and string operators, functions
    (
        "SELECT i = l, ti < dc, dc = 1, i = 1.5, s = i, s < dc, s = dt, dt < ts, ts = tn, s = bo, i = NULL, "
        "NULL = NULL, l <=> s FROM n",
        {},
    ),
    (
        "SELECT i IN (1, 2L), i IN (1.5, 2), s IN (1, 2), dc IN (1, 2.55), i IN (NULL, 1), dt IN ('2020-01-01') FROM n",
        {},
    ),
    ("SELECT i BETWEEN 1 AND 2L, dt BETWEEN '2020-01-01' AND ts, dc BETWEEN 1 AND 2.5, s BETWEEN 1 AND 'x' FROM n", {}),
    (
        "SELECT CASE WHEN bo THEN i ELSE l END, CASE WHEN bo THEN dc ELSE 0 END, CASE WHEN bo THEN s ELSE i END, "
        "CASE WHEN bo THEN NULL ELSE dt END, CASE WHEN bo THEN f ELSE i END FROM n",
        {},
    ),
    (
        "SELECT NOT s, s AND bo, bo OR 'true', NOT NULL, i LIKE 1, dt LIKE '2020%', concat(i, s), concat(bi, bi), "
        "concat(bo, NULL), i || s FROM n",
        {},
    ),
    (
        "SELECT sum(s), avg(i), sum(ti), sum(dc), avg(dw), sum(f), sum(NULL), substring(i, 1, 2), "
        "substring(s, '1', 2L), upper(dt), coalesce(i, l), coalesce(dc, i), coalesce(1, 2.5), coalesce(dt, s) FROM n",
        {},
    ),
    ("SELECT i IN (SELECT l FROM n), s IN (SELECT i FROM n), dc IN (SELECT i FROM n) FROM n", {}),
    ("SELECT * FROM n x FULL JOIN (SELECT l AS i FROM n) y USING (i)", {}),
    ("SELECT i FROM n GROUP BY i HAVING i", {}),
    ("SELECT i FROM n GROUP BY i HAVING 'true'", {}),
    # coercion errors
    ("SELECT ti + bo FROM n", {}),
    ("SELECT s + s FROM n", {}),
    ("SELECT bo * bo FROM n", {}),
    ("SELECT dt / dt FROM n", {}),
    ("SELECT NULL + bo FROM n", {}),
    ("SELECT NULL - s FROM n", {}),
    ("SELECT i = dt FROM n", {}),
    ("SELECT -bo FROM n", {}),
    ("SELECT NOT i FROM n", {}),
    ("SELECT i AND bo FROM n", {}),
    ("SELECT i AND i FROM n", {}),
    ("SELECT i IN (dt, dt) FROM n", {}),
    ("SELECT CASE WHEN bo THEN 1 WHEN i THEN 2 END FROM n", {}),
    ("SELECT CASE WHEN NULL THEN 1 END FROM n", {}),
    ("SELECT CASE WHEN bo THEN 1 WHEN bo THEN dt ELSE s END FROM n", {}),
    ("SELECT coalesce(i, dt, bo) FROM n", {}),
    ("SELECT sum(bo) FROM n", {}),
    ("SELECT substring(s, dt, 2) FROM n", {}),
    ("SELECT * FROM n WHERE i", {}),
    ("SELECT * FROM n WHERE NULL", {}),
    ("SELECT count(*) FILTER (WHERE i) FROM n", {}),
    ("SELECT * FROM n JOIN t ON i", {}),
    ("SELECT i FROM n LIMIT 1L", {}),
    ("SELECT i FROM n LIMIT CAST(1 AS SMALLINT)", {}),
    ("SELECT i IN (SELECT dt FROM n) FROM n", {}),
    ("SELECT (i, s) IN (SELECT i FROM n) FROM n", {}),
    ("SELECT i IN (SELECT i, s FROM n) FROM n", {}),
    ("SELECT (i, s) IN (SELECT i, dt FROM n) FROM n", {}),
    ("SELECT * FROM n x JOIN (SELECT dt AS i FROM n) y USING (i)", {}),
    ("SELECT i NOT IN (1, dt) FROM n", {}),
    ("SELECT i NOT IN (SELECT dt FROM n) FROM n", {}),
    ("SELECT i BETWEEN dt AND 1 FROM n", {}),
    ("SELECT i NOT BETWEEN 1 AND dt FROM n", {}),
    ("SELECT (i, s) NOT IN (SELECT i FROM n) FROM n", {}),
    ("SELECT CAST('1' AS INT), CAST(1 AS DECIMAL(1,0)), CAST(1.55 AS DECIMAL(2,1)), TRY_CAST(1 AS BIGINT) FROM n", {}),
    ("SELECT substring(NULL, 1, 2), upper(NULL), 1Y + dc, 1S * dc, ti * 2Y FROM n", {}),
    ("SELECT substring(s, 1, dt) FROM n", {}),
    ("SELECT CASE WHEN bo THEN 1 WHEN bo THEN 2 WHEN bo THEN 3 WHEN i THEN 4 END FROM n", {}),
    ("SELECT (i + 1) IN (SELECT dt FROM n) FROM n", {}),
    ("SELECT (l, s) IN (SELECT i, s FROM n) FROM n", {}),
    ("SELECT * FROM n x JOIN (SELECT l AS i, s FROM n) y USING (i, s)", {}),
    ("SELECT a FROM t GROUP BY a HAVING a IN (SELECT a FROM u)", {}),
    # subqueries to joins: TPC-H shapes, Spark's other shapes, and what Spark rejects
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a)", {}),
    ("SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND d > b)", {}),
    ("SELECT a FROM t WHERE a IN (SELECT a FROM u WHERE d = b)", {}),
    ("SELECT a FROM t WHERE a NOT IN (SELECT a FROM u)", {}),
    ("SELECT a FROM t WHERE (a, b) NOT IN (SELECT a, d FROM u)", {}),
    ("SELECT a FROM t WHERE a NOT IN (SELECT a FROM u WHERE d = b)", {}),
    ("SELECT (SELECT max(a) FROM u WHERE u.a = t.a) FROM t", {}),
    ("SELECT (SELECT count(*) FROM u WHERE u.a = t.a) FROM t", {}),
    ("SELECT a FROM t WHERE a = (SELECT max(a) FROM u WHERE u.d = t.b AND u.a > 1)", {}),
    ("SELECT a FROM t WHERE c = (SELECT sum(a) FROM u WHERE u.a = t.a) + 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(u.a) FROM u WHERE u.d = t.b) IS NULL", {}),
    ("SELECT sum(a) FROM t GROUP BY b HAVING sum(a) > (SELECT max(a) FROM u WHERE u.d = t.b)", {}),
    ("SELECT a, (SELECT max(u.a) FROM u WHERE u.d = t.b) + 1 AS x FROM t ORDER BY x", {}),
    ("SELECT a FROM t WHERE c IN (SELECT max(u.a) FROM u WHERE u.d = t.b)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u)", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u) > 1", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE EXISTS (SELECT 1 FROM n WHERE n.i = u.a))", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE EXISTS (SELECT 1 FROM n WHERE n.i = c))", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE EXISTS (SELECT 1 FROM n WHERE n.i = t.c))", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a ORDER BY t.c)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a) OR c > 1", {}),
    ("SELECT (SELECT a FROM u WHERE u.a = t.a) FROM t", {}),
    ("SELECT (SELECT count(*) + 1 FROM u WHERE u.a = t.a) FROM t", {}),
    ("SELECT a FROM t WHERE a > (SELECT max(a) FROM u WHERE u.a > t.a)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u JOIN n ON u.a = n.i AND n.l = t.c)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT t.a FROM u)", {}),
    (
        "SELECT a FROM t WHERE (SELECT max(u.a) FROM u WHERE u.d = t.b) > 1 AND (SELECT max(u.a) FROM u"
        " WHERE u.d = t.b) < 5",
        {},
    ),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u WHERE CAST(u.a AS BIGINT) = t.c) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u WHERE u.a + 1 = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u WHERE u.a = t.a AND u.d = t.b) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u WHERE u.a > 0 AND u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u WHERE u.a = t.a) BETWEEN 1 AND 5", {}),
    ("SELECT a FROM t WHERE (SELECT count(a) FROM u WHERE u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) - 1 FROM u WHERE u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) + min(a) FROM u WHERE u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT -max(a) FROM u WHERE u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT CAST(max(a) AS BIGINT) FROM u WHERE u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT coalesce(max(a), 0) FROM u WHERE u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a), min(a) FROM u WHERE u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u WHERE u.a = t.a GROUP BY d) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u WHERE u.a = t.a LIMIT 1) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM (SELECT a FROM u) x WHERE x.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(x) FROM (SELECT a AS x FROM u) v WHERE v.x = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a) FROM u WHERE u.a = t.a) IN (SELECT a FROM u)", {}),
    ("SELECT a FROM t WHERE a > (SELECT max(a) FROM u WHERE u.a = t.a AND EXISTS (SELECT 1 FROM n))", {}),
    (
        "SELECT a FROM t WHERE c > 0 AND (SELECT max(a) FROM u WHERE u.a = t.a) > 1 AND a NOT IN (SELECT a"
        " FROM u WHERE u.d = t.b)",
        {},
    ),
    (
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND (SELECT max(i) FROM n WHERE n.i = u.a) > 0)",
        {},
    ),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND EXISTS (SELECT 1 FROM n WHERE n.i = u.a))", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND u.a IN (SELECT i FROM n))", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a OR u.a IN (SELECT i FROM n))", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a ORDER BY u.d)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT * FROM u WHERE u.a = t.a)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT a FROM (SELECT a, d FROM u) x WHERE x.a = t.a)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u x(p, q) WHERE x.p = t.a)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.a = t.a)", {}),
    ("WITH c AS (SELECT a, b FROM t) SELECT a FROM c WHERE EXISTS (SELECT 1 FROM c c2 WHERE c2.a = c.a)", {}),
    ("SELECT a FROM t WHERE a IN (SELECT a FROM t)", {}),
    ("SELECT a FROM t WHERE a IN (SELECT a FROM (SELECT a, d FROM u WHERE d = t.b) x)", {}),
    ("SELECT a FROM t WHERE a IN (SELECT x FROM (SELECT a AS x, d FROM u) v WHERE v.d = t.b)", {}),
    ("SELECT a FROM t WHERE a IN (SELECT a FROM u x(a, d) WHERE x.d = t.b)", {}),
    ("SELECT a FROM t WHERE c IN (SELECT t.c FROM u WHERE u.a = t.a)", {}),
    ("SELECT * FROM t JOIN u ON t.a = (SELECT max(i) FROM n WHERE n.i = u.a)", {}),
    ("SELECT sum((SELECT max(a) FROM u WHERE u.a = t.a)) FROM t", {}),
    ("SELECT count(*) FROM t GROUP BY (SELECT max(a) FROM u WHERE u.a = t.a)", {}),
    ("SELECT a FROM t ORDER BY (SELECT max(u.a) FROM u WHERE u.d = t.b)", {}),
    ("SELECT a FROM t WHERE (SELECT max(u.a + t.a) FROM u WHERE u.a = t.a) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(t.a) FROM u) > 1", {}),
    ("SELECT a, (SELECT max(u.a) + t.a FROM u WHERE u.a = t.a) FROM t", {}),
    ("SELECT a FROM t WHERE (SELECT max(u.a) FROM u WHERE u.a = t.a AND u.a = CAST(t.c AS INT)) > 1", {}),
    ("SELECT a FROM t WHERE (SELECT max(a > 0) FROM u WHERE u.a = t.a)", {}),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a ORDER BY t.c DESC NULLS LAST)", {}),
    ("SELECT EXISTS (SELECT 1 FROM u WHERE u.a = t.a) FROM t", {}),
    ("SELECT a IN (SELECT a FROM u) FROM t", {}),
    ("SELECT a IN (SELECT a FROM u WHERE u.d = t.b) FROM t", {}),
    # Spark resolves these; Krishiv rejects them explicitly for now
    ("SELECT dt + i, dt - dt FROM n", {}),
    ("SELECT s = 1 FROM n", NO_ANSI),
]

# Type coercion matrices over n's columns and some literals: every pair under `+`, `/` and `=`, and every
# cast in each mode.
OPERANDS = ["ti", "si", "i", "l", "f", "d", "dc", "dw", "s", "bo", "dt", "ts", "tn", "bi", "1", "1L", "1.5"]
OPERANDS += ["CAST(1.5 AS DOUBLE)", "'1'", "NULL"]
MATRIX = [(f"SELECT {x} {op} {y} FROM n", {}) for op in ["+", "/", "="] for x in OPERANDS for y in OPERANDS]
CAST_TYPES = ["TINYINT", "SMALLINT", "INT", "BIGINT", "FLOAT", "DOUBLE", "DECIMAL(10,2)", "STRING", "BOOLEAN"]
CAST_TYPES += ["DATE", "TIMESTAMP", "TIMESTAMP_NTZ", "BINARY"]
# TRY_CAST and CAST of non-NULL values (nullability).
LITERALS = ["1Y", "1S", "1", "1L", "CAST(1.5 AS FLOAT)", "1.5D", "CAST(1.5 AS DECIMAL(10,2))", "'1'", "true"]
LITERALS += ["DATE'2020-01-01'", "CAST(DATE'2020-01-01' AS TIMESTAMP)", "X'01'"]
CASTS = [
    (f"SELECT {cast}({x} AS {t}) FROM n", conf)
    for x in LITERALS
    for t in [*CAST_TYPES, "DECIMAL(38,0)"]
    for cast, conf in [("CAST", {}), ("TRY_CAST", {})]
] + [
    (f"SELECT {cast}({x} AS {t}) FROM n", conf)
    for x in OPERANDS
    if not x[0].isdigit()
    for t in CAST_TYPES
    for cast, conf in [("CAST", {}), ("CAST", NO_ANSI), ("TRY_CAST", {})]
]

NO_ANSI_CONF = {"spark.sql.ansi.enabled": "false"}

# Analyzer fixes (U-an3): a name two select items output, SELECT without FROM,
# bare current_date, the void type, coalesce's common type, extract's errors
# and div.
ANALYZER_FIXES = [
    (sql, conf)
    for sql, conf in [
        ("SELECT a, CAST(a AS STRING) AS a FROM t ORDER BY a", {}),
        ("SELECT CAST(a AS STRING) AS a, a FROM t ORDER BY a", {}),
        ("SELECT a, CAST(a AS STRING) AS a FROM t ORDER BY a + 1", {}),
        ("SELECT a, CAST(a AS STRING) AS a FROM (SELECT a FROM t) ORDER BY a", {}),
        ("SELECT 1", {}),
        ("SELECT 1 AS x, 'y' AS z, 2.5, TRUE", {}),
        ("SELECT 1 WHERE 1 = 1", {}),
        ("SELECT 1 LIMIT 1", {}),
        ("SELECT count(*)", {}),
        ("SELECT a", {}),
        ("SELECT (SELECT 1)", {}),
        ("SELECT current_date", {}),
        ("SELECT CURRENT_TIMESTAMP AS ts", {}),
        ("SELECT current_date FROM t", {}),
        ("SELECT a, current_date FROM t ORDER BY current_date", {}),
        ("SELECT 1 AS current_date, current_date FROM t", {}),
        ("SELECT a AS current_date, count(*) FROM t GROUP BY current_date", {}),
        ("SELECT (SELECT current_date) FROM t", {}),
        ("SELECT NULL", {}),
        ("SELECT NULL AS n, 1", {}),
        ("SELECT CAST(NULL AS VOID)", {}),
        ("SELECT CAST(a AS VOID) FROM t", {}),
        ("SELECT TRY_CAST(b AS VOID) FROM t", {}),
        ("SELECT coalesce(NULL, NULL)", {}),
        ("SELECT CASE WHEN TRUE THEN NULL END", {}),
        ("SELECT (SELECT NULL)", {}),
        ("SELECT n FROM (SELECT NULL AS n FROM t) WHERE n IS NULL", {}),
        ("SELECT 1 FROM t WHERE NULL", {}),
        ("SELECT 1 FROM t LIMIT NULL", {}),
        ("SELECT a + NULL, NULL = NULL, -NULL FROM t", {}),
        ("SELECT coalesce(a, b) FROM t", NO_ANSI_CONF),
        ("SELECT coalesce(a, b) FROM t", {}),
        ("SELECT coalesce(c, CAST(a AS FLOAT)) FROM t", NO_ANSI_CONF),
        ("SELECT coalesce(c, CAST(a AS FLOAT)) FROM t", {}),
        ("SELECT coalesce(b, DATE '2020-01-01', a) FROM t", NO_ANSI_CONF),
        ("SELECT coalesce(a, TRUE) FROM t", NO_ANSI_CONF),
        ("SELECT coalesce(b, CAST(a AS BINARY)) FROM t", NO_ANSI_CONF),
        ("SELECT coalesce(CAST(a AS DECIMAL(38,0)), CAST(a AS DECIMAL(10,5))) FROM t", {}),
        ("SELECT coalesce(CAST(a AS DECIMAL(36,0)), CAST(a AS DECIMAL(10,5))) FROM t", NO_ANSI_CONF),
        ("SELECT coalesce(CAST(a AS DECIMAL(38,10)), CAST(a AS DECIMAL(38,2))) FROM t", {}),
        ("SELECT extract(year FROM a) FROM t", {}),
        ("SELECT extract(w FROM a) FROM t", {}),
        ("SELECT extract(dow_iso FROM a) FROM t", {}),
        ("SELECT extract(second FROM a) FROM t", {}),
        ("SELECT extract(hour FROM a) FROM t", {}),
        ("SELECT date_part('doy', a) FROM t", {}),
        ("SELECT extract(epoch FROM a) FROM t", {}),
        ("SELECT a div 2, c div a, 7.5 div 2, a div NULL, b div 2 FROM t", {}),
        ("SELECT CAST(a AS DECIMAL(38,10)) div CAST(a AS DECIMAL(20,0)) FROM t", {}),
        ("SELECT a div 0 + 1 FROM t", {}),
        ("SELECT 1.5D div 2 FROM t", {}),
        ("SELECT b div b FROM t", {}),
        ("SELECT DATE '2020-01-01' div 2", {}),
        ("SELECT CAST(a AS FLOAT) div CAST(a AS FLOAT) FROM t", {}),
        ("SELECT b div 2 FROM t", NO_ANSI_CONF),
        ("SELECT b div b FROM t", NO_ANSI_CONF),
    ]
]

# Constant folding of arguments that need a constant, and ORDER BY / HAVING over a grouping expression
# (`--section folding`, written to its own fixture). Spark folds any foldable expression for these
# arguments (`round`'s scale, `ceil`/`floor`'s scale, `date_part`'s field, `LIMIT`) and raises what
# folding raises; `first`/`last` match a boolean `Literal` and fold nothing.
FOLD_SCALES = [
    "1+1",
    "i",
    "1+i",
    "2147483647+1",
    "CAST(1 AS BIGINT)+1",
    "-(-2)",
    "5/2",
    "5 div 2",
    "5 % 3",
    "1/0",
    "if(true,1,2)",
    "CASE WHEN 1=1 THEN 3 ELSE 2 END",
    "coalesce(NULL,2)",
    "abs(-2)",
    "length('abc')",
    "1.5+1.5",
    "1.5D+1",
    "NULL+1",
    "3+5L",
    "2L",
    "CAST(2 AS INT)",
    "1=1",
    "'1'||'1'",
    "upper('2')",
    "(1+1) * (2-1)",
    "1 < 2",
    "NOT TRUE",
    "CASE WHEN 1=2 THEN 1/0 ELSE 2 END",
    "if(1=1, 2, 1/0)",
]
FOLDING = (
    [
        (f"SELECT {fn}(dc, {x}) FROM n", conf)
        for fn in ["round", "bround", "ceil", "floor"]
        for x in FOLD_SCALES
        for conf in [{}, NO_ANSI]
    ]
    + [
        (f"SELECT date_part({x}, dt) FROM n", conf)
        for x in ["'y'||'ear'", "s", "CAST(NULL AS STRING)", "upper('year')", "1+1"]
        for conf in [{}, NO_ANSI]
    ]
    + [
        (f"SELECT {fn}(i, {x}) FROM n", {})
        for fn in ["first", "last"]
        for x in ["true", "false", "1=1", "true AND true", "i > 1", "NOT FALSE", "NULL", "CAST(true AS BOOLEAN)"]
    ]
    + [
        (f"SELECT {fn}(dt, {x}) FROM n", conf)
        for fn in ["date_add", "date_sub"]
        for x in ["'1'||'2'", "'1'||'.5'", "upper('1')", "1+1", "'a'", "s", "1.5"]
        for conf in [{}, NO_ANSI]
    ]
    + [
        (f"SELECT date_add(dt, {x}) FROM n", NO_ANSI)
        for x in [
            "'1'",
            "'1.5'",
            "' 12 '",
            "'+3'",
            "'1.'",
            "'.5'",
            "'1e2'",
            "'0x1'",
            "'2147483648'",
            "'-0'",
            "'1.0'",
            "'abc'",
            "''",
            "' '",
            "'1 2'",
        ]
    ]
    + [
        (f"SELECT * FROM n LIMIT {x}", conf)
        for x in [
            "1+1",
            "i",
            "2147483647+1",
            "length('ab')",
            "CAST(NULL AS INT)",
            "-1",
            "1+(-2)",
            "1L",
            "'1'",
            "1.5",
            "CASE WHEN 1=1 THEN 5 ELSE 6 END",
            "(SELECT 1)",
        ]
        for conf in [{}, NO_ANSI]
    ]
)
ORDER_BY_GROUPED = [
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY x + 2",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY 1 + x",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY m + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY x",
    "SELECT x m, count(*) c FROM g GROUP BY x ORDER BY x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY abs(x + 1)",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY (x + 1) * count(*)",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY sum(x)",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY sum(x + 1)",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 HAVING x + 1 > 3 ORDER BY x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 HAVING x + 2 > 3",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 HAVING x + 1 > 3",
    "SELECT count(*) c FROM g GROUP BY x + 1 ORDER BY x + 1",
    "SELECT count(*) c FROM g GROUP BY x + 1 HAVING x + 1 > 3",
    "SELECT count(*) c FROM g GROUP BY x + 1 HAVING x + 1 > 3 ORDER BY x + 1",
    "SELECT sum(l) FROM g GROUP BY x + 1 HAVING x + 1 > 3 ORDER BY x + 1 DESC",
    "SELECT count(*) c FROM g GROUP BY x + 1 HAVING x + 1 > 3 ORDER BY x + 2",
    # A HAVING that grows the aggregate leaves a projection over it: a sort cannot grow it again
    "SELECT count(*) c FROM g GROUP BY x HAVING sum(l) > 1 ORDER BY sum(l)",
    "SELECT count(*) c FROM g GROUP BY x HAVING sum(l) > 1 ORDER BY c",
    "SELECT count(*) c FROM g GROUP BY x + 1 HAVING sum(l) > 1 ORDER BY x + 1",
    "SELECT count(*) c FROM g GROUP BY x HAVING x > 1 ORDER BY sum(l)",
    "SELECT count(*) c FROM g GROUP BY x HAVING count(*) > 0 ORDER BY sum(l)",
    "SELECT count(*) c FROM g GROUP BY x + 1 HAVING count(*) > 0 ORDER BY x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY m ORDER BY x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY 1 ORDER BY x + 1",
    "SELECT DISTINCT x + 1 FROM g ORDER BY x + 1",
    "SELECT DISTINCT x + 1 m FROM g ORDER BY x + 2",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY x + 1 DESC, c",
    "SELECT abs(x) m FROM g GROUP BY abs(x) ORDER BY abs(x)",
    "SELECT X + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY X + 1",
    "SELECT g.x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY g.x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY g.x + 1 ORDER BY x + 1",
    "SELECT sum(x) + 1 m FROM g GROUP BY x + 1 ORDER BY x + 1",
    "SELECT sum(x) s FROM g GROUP BY x + 1 ORDER BY x + 1, s",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY x + 1 + 1",
    "SELECT cast(x as string) m, count(*) c FROM g GROUP BY cast(x as string) ORDER BY cast(x as string)",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY (SELECT 1) + x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY CASE WHEN x + 1 > 1 THEN 1 ELSE 2 END",
    "SELECT (x + 1) * 2 m FROM g GROUP BY x + 1 ORDER BY x + 1",
    "SELECT (x + 1) * 2 m FROM g GROUP BY x + 1 ORDER BY (x + 1) * 2",
    "SELECT x + l m, count(*) c FROM g GROUP BY x + l ORDER BY l + x",
    "SELECT x * l m, count(*) c FROM g GROUP BY x * l ORDER BY l * x",
    "SELECT (x = 1 AND l = 2) m, count(*) c FROM g GROUP BY (x = 1 AND l = 2) ORDER BY (l = 2 AND x = 1)",
    "SELECT (x = 1 OR l = 2) m, count(*) c FROM g GROUP BY (x = 1 OR l = 2) ORDER BY (l = 2 OR x = 1)",
    "SELECT (x = l) m, count(*) c FROM g GROUP BY (x = l) ORDER BY (l = x)",
    "SELECT (x <=> l) m, count(*) c FROM g GROUP BY (x <=> l) ORDER BY (l <=> x)",
    "SELECT (x < l) m, count(*) c FROM g GROUP BY (x < l) ORDER BY (l > x)",
    "SELECT (x <= l) m, count(*) c FROM g GROUP BY (x <= l) ORDER BY (l >= x)",
    "SELECT greatest(x, l) m, count(*) c FROM g GROUP BY greatest(x, l) ORDER BY greatest(l, x)",
    "SELECT least(x, l) m, count(*) c FROM g GROUP BY least(x, l) ORDER BY least(l, x)",
    "SELECT x - l m, count(*) c FROM g GROUP BY x - l ORDER BY l - x",
    "SELECT concat(cast(x as string), cast(l as string)) m, count(*) c FROM g GROUP BY concat(cast(x as string), "
    "cast(l as string)) ORDER BY concat(cast(l as string), cast(x as string))",
    "SELECT coalesce(x, l) m, count(*) c FROM g GROUP BY coalesce(x, l) ORDER BY coalesce(l, x)",
    "SELECT (x + 1) + 2 m, count(*) c FROM g GROUP BY (x + 1) + 2 ORDER BY x + (1 + 2)",
    "SELECT x + 1 + l m, count(*) c FROM g GROUP BY x + 1 + l ORDER BY l + 1 + x",
    "SELECT x + l m, count(*) c FROM g GROUP BY l + x ORDER BY x + l",
    "SELECT l - l % 60 m, count(*) c FROM g WHERE x = 62 GROUP BY l - l % 60 ORDER BY l - l % 60 LIMIT 10",
    "SELECT x m, count(*) c FROM g GROUP BY x HAVING x > 3 ORDER BY x",
    "SELECT x m, count(*) c FROM g GROUP BY x HAVING x > 3 ORDER BY sum(l)",
    "SELECT x m, count(*) c FROM g GROUP BY x HAVING count(*) > 3 ORDER BY sum(l)",
    "SELECT x m FROM g GROUP BY x HAVING count(*) > 3 ORDER BY x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 HAVING count(*) > 3 ORDER BY x + 1",
    "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 HAVING count(*) > 3 ORDER BY m",
]
FOLDING += [(sql, conf) for sql in ORDER_BY_GROUPED for conf in [{}]]

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


def view_query(cols: str) -> str:
    """`SELECT CAST(NULL AS <type>) AS <name>, ... WHERE false` for a DDL column list."""
    parts, depth, cur = [], 0, ""
    for ch in cols:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    items = []
    for part in parts:
        name, type_ = part.split(" ", 1)
        items.append(f"CAST(NULL AS {type_}) AS {name}")
    return f"SELECT {', '.join(items)} WHERE false"


def main() -> None:
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/analyzer-goldens.json")
    ap.add_argument(
        "--section",
        choices=["all", "folding"],
        default="all",
        help="`folding` writes only the constant-folding and ORDER BY-over-aggregate cases "
        "(tests/fixtures/spark-4.2.0/analyzer-folding-goldens.json)",
    )
    ap.add_argument(
        "--views",
        action="store_true",
        help="create the tables as temporary views over empty relations (no catalog writes: for a shared server)",
    )
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    if spark.version != "4.2.0":
        raise SystemExit(f"expected Spark 4.2.0, found {spark.version}")
    if args.views:
        for name, cols in TABLES.items():
            spark.sql(f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {view_query(cols)}")
    else:
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
    plan_ids = []
    if args.section == "folding":
        corpus = [{"sql": sql, "conf": conf, **run_sql(sql, conf)} for sql, conf in FOLDING]
    else:
        for path in sorted(TPCH_DIR.glob("q*.sql")):
            sql = path.read_text(encoding="utf-8").strip().rstrip(";")
            tpch.append({"id": path.stem, **run_sql(sql, {})})
        corpus = [
            {"sql": sql, "conf": conf, **run_sql(sql, conf)} for sql, conf in CORPUS + MATRIX + CASTS + ANALYZER_FIXES
        ]
        plan_ids = [{"id": i, "dataframe": d, **analyze(f)} for i, d, f in plan_id_cases(spark)]
    for name in TABLES:
        spark.sql(f"DROP VIEW IF EXISTS {name}" if args.views else f"DROP TABLE IF EXISTS {name}")

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
