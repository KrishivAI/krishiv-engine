#!/usr/bin/env python3
"""Generate the Spark SQL parse-probe fixture (P2-T070).

Every expectation comes from a REAL Spark 4.2 run, never from docs or memory: each statement below is
parsed by Spark's session SQL parser (`sqlParser.parsePlan`, what `spark.sql` uses) inside the Spark
container (`sqlparse/parse_probe.scala`, run through `spark-shell`) with the case's configs set, and the
fixture records whether it parsed or the error condition, SQLSTATE, message parameters and query-context
indexes Spark raised. The cases are the places where Spark's ANTLR grammar resolves keyword/identifier
ambiguity by prediction (keywords as columns and aliases, trailing keywords), error positions and
`extra input` / `missing` hints, the keyword configs, and lexer edge cases; type strings go through the
data-type, table-schema and `DataType.fromDDL` entry points (P2-T078). Output is deterministic and
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
    # hints (P2-T078)
    "SELECT /*+ IDENTIFIER('BROADCAST')(hint_test) */ * FROM hint_test",
    "SELECT /*+ BROADCAST(t) REPARTITION(3) */ * FROM t",
    "SELECT /*+ BROADCAST(t),, */ * FROM t",
    "SELECT /*+ */ * FROM t",
    "SELECT /*+ BROADCAST( */ * FROM t",
    "SELECT /*+ 1 */ * FROM t",
    "SELECT /*+ a.b */ * FROM t",
    "SELECT /*+ COALESCE(1+1) */ * FROM t",
    "SELECT /*+ BROADCAST(t) */ /*+ MERGE(t) */ * FROM t",
    "SELECT /*+ x('a', 1, t.c) */ * FROM t",
    "SELECT /*+ x(-1) */ * FROM t",
    "SELECT /*+ x(a b) */ * FROM t",
    "SELECT /*+ x() */ * FROM t",
    "SELECT /*+ x(a, */ * FROM t",
    "SELECT /*+ x(a)) */ * FROM t",
    "SELECT /*+ x y */ * FROM t",
    "SELECT /*+ x(1) 2 */ 1",
    "SELECT /*+ x(f(1)) */ 1",
    "SELECT /*+ x(INTERVAL 1 DAY) */ 1",
    "SELECT /*+ x(@a) */ 1",
    "SELECT /*+ x(?) */ 1",
    "SELECT /*+ select */ 1",
    "SELECT /*+ x(a[0]) */ 1",
    "SELECT /*+ x(CAST(1 AS INT)) */ 1",
    "SELECT /*+ x((1)) */ 1",
    "SELECT DISTINCT /*+ x */ 1",
    "SELECT /*+ x */ DISTINCT 1",
    "SELECT /*+ IDENTIFIER('x') */ 1",
    # data types inside statements (P2-T078)
    "SELECT CAST(x AS array<int)",
    "SELECT CAST(x AS struct<a int,>)",
    "SELECT CAST(x AS map<int>)",
    "SELECT CAST(x AS array<int>>)",
    "SELECT CAST(x AS decimal(10,2)",
    "SELECT CAST(x AS int x)",
    "SELECT CAST(x AS foo x)",
    "SELECT CAST(x AS string x y)",
    "SELECT CAST(x AS int",
    "SELECT x::array<int",
    "SELECT x::array<int> y z",
    "SELECT CAST(x AS `int`)",
    "SELECT CAST(x AS array<foo>)",
    "SELECT CAST(x AS struct)",
    "SELECT CAST(x AS varchar)",
    "SELECT CAST(x AS int(1))",
    "SELECT CAST(x AS decimal(5,6))",
    "SELECT CAST(x AS time(10))",
    "SELECT x::foo",
    "SELECT CAST(CAST(x AS foo) AS bar)",
    "SELECT CAST(x AS struct<a:foo, b:bar>)",
    # nesting deeper than the parser's stack (P2-T076): Spark reports the whole statement
    "SELECT 1" + " + 1" * 1000,
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
# Data type strings (P2-T078): `CatalystSqlParser.parseDataType` and `DataType.fromDDL`.
TYPE_STRINGS = [
    "a InvalidType",
    "InvalidType",
    "int",
    "a int",
    "a int, b string",
    "struct<a:int>",
    "STRUCT<a: INT, b: ARRAY<STRING>>",
    "Array < Int >",
    "array<int>/*c*/",
    "array/**/<int>",
    "int\n",
    "",
    "/* x",
    # syntax errors: the `<` of a complex type that does not parse, hints after a type
    "array<int",
    "array<int>>",
    "array<>",
    "array<int,int>",
    "array<array<int>",
    "array<struct<a:int>",
    "array<int NOT NULL>",
    "map<int>",
    "map<int,>",
    "map<,int>",
    "map<int, string x>",
    "struct<a int,>",
    "struct<a:int b:int>",
    "struct<a.b:int>",
    "struct<a:int COMMENT 1>",
    "struct<a: int NOT NULL, b: int COMMENT 1>",
    "struct<a: int COMMENT 'x', b: int COMMENT 1>",
    "struct<a int NOT NULL x>",
    "struct<a: decimal(10, 2) x>",
    "struct<:int>",
    "array<int> x",
    "array<int>> x",
    "array<int> > 1",
    "array<int>)",
    "struct<a:int> x",
    "int x",
    "int x y",
    "int (",
    "int )",
    "int ,",
    "int;",
    "int(1) x",
    "string x",
    "date x",
    "timestamp x",
    "timestamp without",
    "timestamp with local",
    "timestamp with time zone x",
    "time(6) without",
    "time(6) with time zone",
    "boolean (",
    "double precision",
    "decimal x",
    "decimal (",
    "decimal(10,2) x",
    "decimal(10,2",
    "decimal(any)",
    "char (",
    "char(3) x",
    "foo x",
    "foo (",
    "foo )",
    "foo(1) x",
    "foo(any)",
    "int(any)",
    "`int` x",
    "`array`<int>",
    "geometry x",
    "geometry(x)",
    "interval x y",
    "interval (",
    "interval day x",
    "interval day to",
    "interval day(1)",
    "interval second to year",
    "interval hour to day",
    "interval day to day",
    "interval month to year",
    "string collate",
    # checks on the type built (`AstBuilder`)
    "foo",
    "foo(1)",
    "`int`",
    "`a`(1)",
    "int(1)",
    "int(1,2)",
    "string(3)",
    "timestamp_ntz(3)",
    "binary(1)",
    "void(1)",
    "interval(1)",
    "array(1)",
    "array",
    "map",
    "struct",
    "struct<>",
    "array<array>",
    "map<int,map>",
    "struct<a:struct>",
    "array<foo>",
    "map<foo,int>",
    "struct<a:foo>",
    "array<char>",
    "char",
    "character",
    "varchar",
    "char(3)",
    "char(3,4)",
    "varchar(0)",
    "decimal(40,0)",
    "array<decimal(40,0)>",
    "numeric(5,6)",
    "decimal(0)",
    "decimal(38,38)",
    "time",
    "time(0)",
    "time(7)",
    "geometry",
    "geometry(any)",
    "geography(4326)",
    "interval",
    "interval year",
    "interval day to second",
    "void",
    "variant",
    "timestamp_ltz",
    "map<array<int>,int>",
    "struct<a:void>",
    "struct<`a b`:int>",
    "struct<a:int NOT NULL>",
    "struct<a:int COMMENT 'x'>",
    "a decimal(40,0)",
    "a char",
    "a foo",
    "`a b` int",
]
# Table schemas (P2-T078): `CatalystSqlParser.parseTableSchema` and `DataType.fromDDL`.
TABLE_SCHEMAS = [
    "a int",
    "a int, b string",
    "a InvalidType",
    "a array<int>, b map<int,string>",
    "a INT NOT NULL",
    "a int NOT NULL COMMENT 'x', b string",
    "a int COMMENT 'x'",
    "a int, a string",
    "`a b` int",
    "select int",
    "a interval day",
    "a struct<b: int>",
    "int",
    "int x",
    "struct<a:int>",
    "a int x",
    "a int b string",
    "a int, b",
    "a int,",
    "a int, ",
    "a int,,b int",
    "a,b int",
    "a:int",
    "a.b int",
    "`a`.b int",
    "a-b int",
    "\u00e9 int",
    "a int COMMENT",
    "a int NOT",
    "a int NOT, b int",
    "a int COMMENT 'x' NOT NULL",
    "a int default 1",
    "a foo x",
    "a foo x, b int",
    "a char",
    "a foo",
    "a foo, b bar",
    "a struct<b:foo>",
    "a decimal(40,0)",
    "a int /* x",
]

PLAN_CASES = (
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
CASES = [(sql, conf, "plan") for sql, conf in PLAN_CASES] + [
    (sql, {}, mode)
    for sqls, modes in [(TYPE_STRINGS, ["datatype", "fromddl"]), (TABLE_SCHEMAS, ["tableschema", "fromddl"])]
    for sql in sqls
    for mode in modes
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="krishiv-bench-spark")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/sql-parse-probe.json")
    args = ap.parse_args()

    data = "".join(json.dumps({"sql": sql, "conf": conf, "mode": mode}) + "\n" for sql, conf, mode in CASES)
    script = (Path(__file__).parent / "sqlparse" / "parse_probe.scala").read_text(encoding="utf-8")
    lines = spark_shell(args.container, script, data)
    version = next(line.split(" ", 1)[1] for line in lines if line.startswith("@@VERSION "))
    results = [json.loads(line[4:]) for line in lines if line.startswith("@@P ")]
    if [r["sql"] for r in results] != [sql for sql, _, _ in CASES]:
        raise SystemExit("spark-shell output does not match the inputs")
    for r, (_, conf, mode) in zip(results, CASES, strict=True):
        r["conf"] = conf
        if mode != "plan":
            r["mode"] = mode
    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": version,
        "image": container_image(args.container),
        "generator": GENERATOR,
        "notes": "Statements parsed by Spark's session SQL parser with each case's configs (a `mode` case "
        "uses the data-type, table-schema or DataType.fromDDL entry point instead); errors carry Spark's "
        "condition, SQLSTATE, message parameters, query-context start/stop indexes, and a ParseException's "
        "line and position.",
        "cases": results,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(results)} cases against Spark {version} -> {args.out}")


if __name__ == "__main__":
    main()
