#!/usr/bin/env python3
"""Generate the Spark SQL parse-probe fixture (P2-T070).

Every expectation comes from a REAL Spark 4.2 run, never from docs or memory: each statement below is
parsed by Spark's session SQL parser (`sqlParser.parsePlan`, what `spark.sql` uses) inside the Spark
container (`sqlparse/parse_probe.scala`, run through `spark-shell`) with the case's configs set, and the
fixture records whether it parsed or the error condition, SQLSTATE, message parameters and query-context
indexes Spark raised. The cases are the places where Spark's ANTLR grammar resolves keyword/identifier
ambiguity by prediction (keywords as columns and aliases, trailing keywords), error positions and
`extra input` / `missing` hints, the keyword configs, and lexer edge cases. Output is deterministic and
contains no local paths or hostnames.

Usage (from repo root; take the bench lock, start Spark, stop it afterwards; see bench/README.md):

    docker compose -f bench/baselines/docker-compose.yml up -d spark
    flock "$KRISHIV_DATA/.bench.lock" nice -n 19 \
      python3 tests/scripts/gen_sql_parse_probe.py --out tests/fixtures/spark-4.2.0/sql-parse-probe.json
    docker compose -f bench/baselines/docker-compose.yml stop spark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gen_function_matrix import container_image, spark_shell
from gen_type_strings import LOCAL_PATH

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_sql_parse_probe.py"
ANSI_KEYWORDS = {"spark.sql.ansi.enabled": "true", "spark.sql.ansi.enforceReservedKeywords": "true"}
DOUBLE_QUOTED = {"spark.sql.ansi.enabled": "true", "spark.sql.ansi.doubleQuotedIdentifiers": "true"}
LEGACY_SETOPS = {"spark.sql.legacy.setopsPrecedence.enabled": "true"}

DEFAULT = [
    # keywords as columns, aliases and table names (ANTLR prediction)
    "SELECT FROM",
    "SELECT all, b FROM t",
    "SELECT all, distinct FROM t",
    "SELECT all + 1",
    "SELECT count(ALL)",
    "SELECT * FROM from t",
    "SELECT * FROM t LATERAL VIEW explode(x) AS c",
    "SELECT a NOT",
    "SELECT a FROM",
    "SELECT a WHERE",
    "SELECT a, FROM",
    "SELECT a FROM t WHERE",
    "SELECT a AS",
    "SELECT * FROM t AS",
    "SELECT a FROM t LIMIT",
    "SELECT * FROM VALUES",
    "SELECT INTERVAL 1 DAY TO",
    "SELECT 1 AS select",
    "SELECT x -> 1",
    "SELECT transform(a, all -> all * all)",
    "SELECT current_date",
    "SELECT a FROM t ORDER BY ALL",
    "SELECT a FROM t GROUP BY ALL",
    "SELECT EXISTS(1)",
    "SELECT a IN ((SELECT 1))",
    "SELECT 1.5a",
    "SELECT a.b.c.* FROM t",
    "SELECT * EXCEPT (a) FROM t",
    "SELECT date_add(d, 1), datediff(year, 1)",
    "FROM t",
    "FROM t SELECT a SELECT b",
    "SELECT a FROM t GROUP BY a, b GROUPING SETS ((a), (a + b) * 2)",
    "SELECT a b, c AS d, e AS (f, g) FROM t",
    "SELECT a (b, c) FROM t",
    "SELECT a FROM t LIMIT ALL",
    "SELECT 1 UNION SELECT 2 INTERSECT DISTINCT SELECT 3",
    "SELECT * FROM t, LATERAL (SELECT t.a) AS s (b)",
    "SELECT * FROM a LEFT SEMI JOIN b SEMI JOIN c ANTI JOIN d",
    "SELECT x[0], x.y.z, x::int, -1::string",
    "SELECT \"a\" \"b\", 'c' 'd'",
    # syntax errors: position and hint
    "SELECT 1 abc def",
    "SELECT (1 AS a)",
    "SELECT 'a\nb' c d e",
    "SELECT\t'unterminated",
    "SELECT 1 +",
    "SELECT (1",
    "-- only a comment",
    "SELECT 1 FROM t WHERE a = 1 b",
    "SELECT array(1) >> 1",
    "SELECT 1 FROM t t2 t3",
    "SELECT a FROM t ORDER BY",
    "SELECT a, count(*) FROM t GROUP BY a HAVING",
    "SELECT (SELECT 1",
    "SELECT 1; SELECT 2",
    "SELECT a FROM t UNION",
    "SELECT a b c",
    "SELECT a.b.c d.e",
    "SELECT f(",
    "SELECT CAST(1 AS)",
    "SELECT CAST(1 AS INT",
    "SELECT a FROM t JOIN",
    "SELECT a FROM t JOIN u ON",
    "SELECT 1 1",
    "SELECT `a",
    "SELECT 'a",
    'SELECT "a',
    "SELECT x::",
    "WITH a AS (SELECT 1)",
    "WITH a AS SELECT 1 SELECT 1",
    "SELECT INTERVAL 1",
    "SELECT a[1",
    "SELECT * FROM test-table",
    "SELECT café FROM t",
    "SELECT 1 /* x",
    "   ",
    # keywords where only an identifier fits
    "WITH order AS (SELECT 1) SELECT * FROM order",
    "WITH select AS (SELECT 1) SELECT 1",
    "SELECT * FROM order",
    "SELECT * FROM t PIVOT (sum(a) FOR order IN (1))",
    "SELECT * FROM t UNPIVOT (order FOR n IN (a))",
    "SELECT a FROM t WINDOW order AS ()",
    "SELECT CASE END",
    "SELECT CASE",
    "SELECT * FROM t AS union",
    "SELECT * FROM t AS join",
    "SELECT 1 AS union",
    # time travel and table options
    "SELECT * FROM t VERSION AS OF 1",
    "SELECT * FROM t TIMESTAMP AS OF '2020-01-01'",
    "SELECT * FROM t FOR SYSTEM_TIME AS OF '2020-01-01'",
    "SELECT * FROM t FOR SYSTEM_VERSION AS OF 'v1' x",
    "SELECT * FROM t WITH ('k' = 'v', a.b 1, c = true)",
    "SELECT * FROM t WITH (k)",
    "SELECT * FROM t WITH (k = x)",
    # IN lists and subqueries
    "SELECT a IN ((SELECT 1), 2)",
    "SELECT a IN ((SELECT 1) + 1)",
    "SELECT a IN ((SELECT 1) UNION (SELECT 2))",
    # facts post-parse validation needs
    "SELECT a ! IN (1)",
    "SELECT a IS ! NULL",
    "SELECT a ! BETWEEN 1 AND 2",
    "SELECT a ! LIKE 'x'",
    "SELECT dateadd(MICROSECOND, 1, t)",
    "SELECT dateadd('MICROSECOND', 1, t)",
    "SELECT interval '10-9' year to month '2-1' year to month",
    "SELECT interval 1 day '2-1' year to month",
    "SELECT interval '1' day to hour 2 hours",
    "SELECT `date` '2020-01-01'",
    "SELECT GEO 'x'",
    # minor grammar points
    "SELECT /*+ BROADCAST(t), */ * FROM t",
    "SELECT EXISTS (SELECT 1) = true",
    "SELECT NOT EXISTS (SELECT 1) AND true",
    "SELECT 1 + NOT a = b",
    "SELECT a * ! b",
    "SELECT a LIKE ANY ('a') ESCAPE 'x'",
    "SELECT sum(a) OVER (CLUSTER BY a ORDER BY b)",
    "SELECT sum(a) OVER (CLUSTER BY a)",
    "SELECT \u216b",
    "SELECT r'abc",
    "SELECT $$x$$",
    "SELECT 1 'secret'",
    "FROM t SELECT 1 SELECT 2",
    # EXPLAIN (P2-T093a)
    "EXPLAIN SELECT 1",
    "explain extended SELECT a FROM t WHERE a > 0",
    "EXPLAIN FORMATTED SELECT 1",
    "EXPLAIN CODEGEN SELECT 1",
    "EXPLAIN COST SELECT 1",
    "EXPLAIN LOGICAL SELECT 1",
    "EXPLAIN LOGICAL SELECT (",
    "EXPLAIN ANALYZE SELECT 1",
    "EXPLAIN",
    "EXPLAIN EXTENDED",
    "EXPLAIN SELECT",
    "EXPLAIN EXTENDED EXTENDED SELECT 1",
    "EXPLAIN SELECT 1;",
    "EXPLAIN (SELECT 1)",
    "EXPLAIN WITH x AS (SELECT 1) SELECT * FROM x",
    "EXPLAIN FROM t SELECT a",
    "EXPLAIN VALUES (1)",
    "EXPLAIN TABLE t",
    "EXPLAIN EXPLAIN SELECT 1",
    "EXPLAIN SELECT 1 abc def",
    "EXPLAIN SELECT 1; SELECT 2",
    "EXPLAIN EXTENDED (SELECT 1) UNION SELECT 2",
    "EXPLAIN extended",
    "SELECT explain FROM t",
    "SELECT * FROM explain",
    "SELECT extended, formatted, codegen, cost, logical FROM t",
    "EXPLAIN a",
    "EXPLAIN COST",
    "EXPLAIN SELECT 1 /* x",
    "EXPLAIN LOGICAL SELECT 1 /* x",
    "EXPLAIN LOGICAL SELECT 1;",
    "EXPLAIN EXPLAIN LOGICAL SELECT 1",
    "EXPLAIN LOGICAL SELECT a ! IN (1)",
    "SELECT a ! IN (1) /* x",
    "EXPLAIN ANALYZE",
    "EXPLAIN ANALYZE TABLE t COMPUTE STATISTICS",
]
CASES = (
    [(sql, {}) for sql in DEFAULT]
    + [
        (sql, ANSI_KEYWORDS)
        for sql in [
            "SELECT 1 AS select",
            "SELECT all FROM t",
            "SELECT a FROM",
            "SELECT current_date",
            "SELECT x -> 1",
            "SELECT end FROM t",
            "SELECT 1 AS end",
            "SELECT * FROM t end",
            "WITH end AS (SELECT 1) SELECT 1",
            "SELECT any(true)",
            "SELECT some(true)",
            "SELECT left('a', 1)",
            "SELECT a FROM t ORDER BY all",
            "SELECT a FROM t GROUP BY all",
            "SELECT * FROM order",
            "EXPLAIN SELECT 1 AS select",
            "SELECT explain FROM t",
        ]
    ]
    + [(sql, {"spark.sql.legacy.bangEqualsNot": "true"}) for sql in ["SELECT a ! IN (1)", "SELECT a IS ! NULL"]]
    + [(sql, DOUBLE_QUOTED) for sql in ['SELECT "a""b" FROM t', "SELECT \"a\" 'b'", 'SELECT 1 "x y" z']]
    + [(sql, LEGACY_SETOPS) for sql in ["SELECT 1 UNION SELECT 2 INTERSECT SELECT 3"]]
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="krishiv-bench-spark")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/sql-parse-probe.json")
    args = ap.parse_args()

    data = "".join(json.dumps({"sql": sql, "conf": conf}) + "\n" for sql, conf in CASES)
    script = (Path(__file__).parent / "sqlparse" / "parse_probe.scala").read_text(encoding="utf-8")
    lines = spark_shell(args.container, script, data)
    version = next(line.split(" ", 1)[1] for line in lines if line.startswith("@@VERSION "))
    results = [json.loads(line[4:]) for line in lines if line.startswith("@@P ")]
    if [r["sql"] for r in results] != [sql for sql, _ in CASES]:
        raise SystemExit("spark-shell output does not match the inputs")
    for r, (_, conf) in zip(results, CASES, strict=True):
        r["conf"] = conf
    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": version,
        "image": container_image(args.container),
        "generator": GENERATOR,
        "notes": "Statements parsed by Spark's session SQL parser with each case's configs; errors carry "
        "Spark's condition, SQLSTATE, message parameters and query-context start/stop indexes.",
        "cases": results,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(results)} cases against Spark {version} -> {args.out}")


if __name__ == "__main__":
    main()
