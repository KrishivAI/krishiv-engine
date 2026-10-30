"""Query texts per dialect: Spark-dialect engines read the DuckDB-exported TPC-DS texts with backtick identifiers."""

from __future__ import annotations

from krishiv_bench.runner import backtick_identifiers, load_queries


def test_double_quoted_identifiers_become_backticks():
    assert backtick_identifiers('SELECT x AS "order count" FROM t') == "SELECT x AS `order count` FROM t"
    # String literals are left alone, '' escapes included.
    sql = "SELECT 'say \"hi\"', 'it''s' AS \"a\""
    assert backtick_identifiers(sql) == "SELECT 'say \"hi\"', 'it''s' AS `a`"
    # A backtick inside a quoted name is doubled; an unterminated quote is left for the engine to reject.
    assert backtick_identifiers('SELECT 1 AS "a`b"') == "SELECT 1 AS `a``b`"
    # DuckDB writes a double quote inside an identifier as "": it stays one quote in the backtick name.
    assert backtick_identifiers('SELECT "a""b" FROM t') == 'SELECT `a"b` FROM t'
    assert backtick_identifiers('SELECT 1 AS """x"""') == 'SELECT 1 AS `"x"`'
    assert backtick_identifiers('SELECT "a""b`c" FROM t') == 'SELECT `a"b``c` FROM t'
    # An unterminated identifier (here ending in an escaped quote) is left for the engine to reject.
    assert backtick_identifiers('SELECT "a"" FROM t') == 'SELECT "a"" FROM t'
    assert backtick_identifiers('SELECT 1 AS "open') == 'SELECT 1 AS "open'
    assert backtick_identifiers("SELECT 'unterminated") == "SELECT 'unterminated"


def test_spark_dialect_engines_get_backticks_and_duckdb_the_original():
    # The eight TPC-DS texts with double-quoted aliases; Spark 4.2.0 rejects them as written (PARSE_SYNTAX_ERROR).
    eight = ["q16", "q32", "q50", "q62", "q92", "q94", "q95", "q99"]
    duck = {q.id: q.sql for q in load_queries("tpcds", "duckdb")}
    for engine in ("krishiv", "spark-connect", "sail"):
        spark = {q.id: q.sql for q in load_queries("tpcds", engine)}
        changed = sorted(q for q in spark if spark[q] != duck[q])
        assert changed == eight
        assert all('"' not in spark[q] for q in eight)
    assert '"order count"' in duck["q16"]
    assert "`order count`" in {q.id: q.sql for q in load_queries("tpcds", "krishiv", ["q16"])}["q16"]
