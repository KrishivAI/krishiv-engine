#!/usr/bin/env python3
"""Generate the window-function golden fixture (P2-T055).

Every expectation comes from a REAL Spark 4.2 run, never from docs or memory. The script defines small
tables as temporary views (each value cast from its text, so the engine under test can build the same
columns), runs each case through `spark.sql` and records the output columns (name, `simpleString` type,
nullability) and rows, or the error condition, SQLSTATE and message parameters Spark raised. Values are
recorded as text: integers in decimal, doubles as Python's `repr` (`nan`, `inf`), decimals in plain
notation with their scale, dates in ISO form, strings as they are, NULL as `null`. Rows are compared as multisets
(ties in an
`ORDER BY` order rows nondeterministically in Spark too). Output is deterministic and contains no local
paths or hostnames.

Usage (from repo root), against a Spark 4.2.0 Spark Connect server (temporary views only; nothing
outlives the session):

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \\
      python tests/scripts/gen_window_goldens.py --remote sc://127.0.0.1:15012 \\
      --out tests/fixtures/spark-4.2.0/window-goldens.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
from decimal import Decimal
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_window_goldens.py"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")

NO_ANSI = {"spark.sql.ansi.enabled": "false"}
M38 = "9" * 38

# name -> (columns, rows); every value is the text the column is cast from (None for NULL).
TABLES: dict[str, tuple[str, list[tuple[str | None, ...]]]] = {
    "t": (
        "p INT, o INT, v INT",
        [
            ("1", "1", "10"),
            ("1", "1", "20"),
            ("1", "2", None),
            ("1", None, "5"),
            ("2", "3", "7"),
            (None, "4", "1"),
            (None, None, "2"),
        ],
    ),
    # TPC-DS-like: decimal(7,2) prices, doubles, strings.
    "e": (
        "k STRING, c STRING, d DECIMAL(7,2), x DOUBLE, s STRING",
        [
            ("a", "x", "10.25", "0.1", "pear"),
            ("a", "x", "20.50", "0.2", "apple"),
            ("a", "y", None, "0.3", None),
            ("a", "y", "3.33", None, "fig"),
            ("b", "x", "99999.99", "1.0E10", "kiwi"),
            ("b", "x", "-0.01", "-1.0E10", "Kiwi"),
            ("b", "z", "0.00", "1.5", "éclair"),
            (None, "x", "1.11", "2.5", "plum"),
        ],
    ),
    # Decimal sums that leave and come back into range.
    "m": ("i INT, x DECIMAL(38,0)", [("1", M38), ("2", "1"), ("3", "-5")]),
    "m2": ("i INT, x DECIMAL(38,0)", [("1", M38), ("2", "-5"), ("3", "1")]),
    "l": ("i INT, x BIGINT", [("1", "9223372036854775807"), ("2", "1"), ("3", "-5")]),
    "f": (
        "i INT, x DOUBLE",
        [("1", "NaN"), ("2", "NaN"), ("3", "0.0"), ("4", "-0.0"), ("5", None), ("6", "1.0")],
    ),
    "r": ("a BIGINT", [(str(i),) for i in range(7)]),
    "one": ("a INT", [("1",)]),
    "dt": ("d DATE, x INT", [("2024-01-31", "1"), ("2024-02-29", "2"), (None, "3"), ("2024-02-29", "4")]),
}

# (id, sql, conf). Rows are compared as multisets.
CASES: list[tuple[str, str, dict[str, str]]] = [
    # ---- ranking functions ----
    (
        "ranking",
        "SELECT p, o, v, row_number() OVER (PARTITION BY p ORDER BY o, v) rn, rank() OVER (PARTITION BY p ORDER BY"
        " o) rk,"
        " dense_rank() OVER (PARTITION BY p ORDER BY o) dr, percent_rank() OVER (PARTITION BY p ORDER BY o) pr,"
        " cume_dist() OVER (PARTITION BY p ORDER BY o) cd, ntile(2) OVER (PARTITION BY p ORDER BY o, v) nt FROM t",
        {},
    ),
    (
        "ranking_names",
        "SELECT rank() OVER (PARTITION BY p ORDER BY o), dense_rank() OVER (ORDER BY o DESC NULLS FIRST),"
        " percent_rank() OVER (ORDER BY o), cume_dist() OVER (ORDER BY o), row_number() OVER (ORDER BY o, p, v) FROM t",
        {},
    ),
    ("ranking_desc", "SELECT p, o, v, rank() OVER (PARTITION BY p ORDER BY o DESC NULLS LAST, v) r FROM t", {}),
    ("ranking_multi_key", "SELECT k, c, d, rank() OVER (ORDER BY k NULLS LAST, c DESC) r FROM e", {}),
    ("rank_args_ignored", "SELECT rank(v) OVER (ORDER BY o) FROM t", {}),
    (
        "ntile",
        "SELECT a, ntile(3) OVER (ORDER BY a) n3, ntile(4) OVER (ORDER BY a) n4, ntile(10) OVER (ORDER BY a) n10,"
        " ntile(1) OVER (ORDER BY a) n1, ntile(7) OVER (ORDER BY a) n7 FROM r",
        {},
    ),
    ("ntile_default", "SELECT ntile() OVER (ORDER BY a) FROM r", {}),
    ("ntile_partitioned", "SELECT a, ntile(2) OVER (PARTITION BY a % 3 ORDER BY a) n FROM r", {}),
    (
        "single_row",
        "SELECT percent_rank() OVER (ORDER BY a), cume_dist() OVER (ORDER BY a), ntile(5) OVER (ORDER BY a) FROM one",
        {},
    ),
    ("rank_by_expr", "SELECT a, rank() OVER (ORDER BY a % 3) r, sum(a) OVER (ORDER BY a % 3) s FROM r", {}),
    (
        "rank_literal_order",
        "SELECT row_number() OVER (ORDER BY 1), rank() OVER (ORDER BY 'a'), sum(v) OVER (PARTITION BY 1 + 1) FROM t",
        {},
    ),
    ("ntile_folded", "SELECT a, ntile(1 + 1) OVER (ORDER BY a) FROM r", {}),
    ("count_null", "SELECT count(NULL) OVER (), count(v) OVER (PARTITION BY p ORDER BY o) FROM t", {}),
    ("empty_input", "SELECT rank() OVER (ORDER BY v), sum(v) OVER () FROM t WHERE v > 1000", {}),
    (
        "union_of_windows",
        "SELECT p, rank() OVER (ORDER BY v) FROM t UNION ALL SELECT p, rank() OVER (ORDER BY o) FROM t",
        {},
    ),
    ("window_over_join", "SELECT a.p, a.v, sum(b.v) OVER (PARTITION BY a.p) FROM t a JOIN t b ON a.p = b.p", {}),
    (
        "window_over_limit",
        "SELECT v, rank() OVER (ORDER BY v) FROM (SELECT v FROM t ORDER BY v NULLS LAST LIMIT 4)",
        {},
    ),
    ("limit_over_window", "SELECT v, rank() OVER (ORDER BY v) r FROM t ORDER BY r, v LIMIT 3", {}),
    (
        "date_keys",
        "SELECT d, x, rank() OVER (ORDER BY d) r, sum(x) OVER (PARTITION BY d) s, min(d) OVER (ORDER BY x) m FROM dt",
        {},
    ),
    ("partition_by_expr", "SELECT p, v, row_number() OVER (PARTITION BY p + 1 ORDER BY v * 2 DESC) FROM t", {}),
    # ---- NaN, -0.0 and NULL keys ----
    ("float_keys_rank", "SELECT x, rank() OVER (ORDER BY x) r, dense_rank() OVER (ORDER BY x) d FROM f", {}),
    (
        "float_keys_partition",
        "SELECT x, sum(1) OVER (PARTITION BY x) n, count(*) OVER (PARTITION BY x ORDER BY i) c FROM f",
        {},
    ),
    ("float_min_max", "SELECT x, max(x) OVER () mx, min(x) OVER (ORDER BY x) mn FROM f", {}),
    # ---- aggregates and default frames ----
    (
        "default_frames",
        "SELECT p, o, v, sum(v) OVER (PARTITION BY p) a, sum(v) OVER (PARTITION BY p ORDER BY o) b,"
        " sum(v) OVER (ORDER BY o, p, v ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) c FROM t",
        {},
    ),
    (
        "aggregate_names",
        "SELECT sum(v) OVER (PARTITION BY p), avg(v) OVER (), count(v) OVER (), count(*) OVER (),"
        " min(v) OVER (PARTITION BY p), max(v) OVER (PARTITION BY p ORDER BY o) FROM t",
        {},
    ),
    (
        "explicit_frames",
        "SELECT p, o, v,"
        " sum(v) OVER (ORDER BY o, p, v RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) a,"
        " sum(v) OVER (ORDER BY o ROWS UNBOUNDED PRECEDING) b0,"
        " sum(v) OVER (PARTITION BY p ORDER BY o RANGE UNBOUNDED PRECEDING) c,"
        " sum(v) OVER (PARTITION BY p ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) d,"
        " sum(v) OVER (PARTITION BY p ORDER BY o RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) e FROM t",
        {},
    ),
    (
        "running_peers",
        "SELECT p, o, v, sum(v) OVER (PARTITION BY p ORDER BY o) s_range,"
        " sum(v) OVER (PARTITION BY p ORDER BY o, v ROWS UNBOUNDED PRECEDING) s_rows,"
        " count(v) OVER (PARTITION BY p ORDER BY o) c, avg(v) OVER (PARTITION BY p ORDER BY o) a FROM t",
        {},
    ),
    (
        "first_last",
        "SELECT p, o, v, first(v) OVER (PARTITION BY p ORDER BY o, v) f, last(v) OVER (PARTITION BY p ORDER BY o) l,"
        " first(v) IGNORE NULLS OVER (ORDER BY o, p, v) fi, last_value(v, true) OVER (PARTITION BY p) li,"
        " first_value(v) OVER (PARTITION BY p ORDER BY v NULLS LAST) fv FROM t",
        {},
    ),
    (
        "filter_clause",
        "SELECT p, sum(v) FILTER (WHERE v > 5) OVER (PARTITION BY p) a,"
        " count(*) FILTER (WHERE o IS NOT NULL) OVER (PARTITION BY p ORDER BY o) b FROM t",
        {},
    ),
    (
        "strings",
        "SELECT k, s, min(s) OVER (PARTITION BY k) mn, max(s) OVER () mx, max(s) OVER (PARTITION BY k ORDER BY s) r"
        " FROM e",
        {},
    ),
    (
        "decimals",
        "SELECT k, c, d, sum(d) OVER (PARTITION BY k) s, avg(d) OVER (PARTITION BY k) a,"
        " sum(d) OVER (PARTITION BY k ORDER BY c, d) rs, avg(d) OVER (PARTITION BY k ORDER BY c) ra,"
        " min(d) OVER (ORDER BY d) mn, max(d) OVER (PARTITION BY c) mx FROM e",
        {},
    ),
    (
        "doubles",
        "SELECT k, c, x, sum(x) OVER (PARTITION BY k ORDER BY c, x ROWS UNBOUNDED PRECEDING) rs,"
        " avg(x) OVER (PARTITION BY k) a, sum(x) OVER (ORDER BY k, c, x) s FROM e",
        {},
    ),
    ("decimal_overflow_running", "SELECT i, sum(x) OVER (ORDER BY i ROWS UNBOUNDED PRECEDING) s FROM m", {}),
    ("decimal_overflow_whole", "SELECT i, sum(x) OVER () s FROM m", {}),
    ("decimal_overflow_whole_2", "SELECT i, sum(x) OVER () s FROM m2", {}),
    (
        "decimal_overflow_running_noansi",
        "SELECT i, sum(x) OVER (ORDER BY i ROWS UNBOUNDED PRECEDING) s FROM m",
        NO_ANSI,
    ),
    ("decimal_avg_overflow", "SELECT i, avg(x) OVER (ORDER BY i ROWS UNBOUNDED PRECEDING) s FROM m", {}),
    ("decimal_avg_overflow_noansi", "SELECT i, avg(x) OVER (ORDER BY i ROWS UNBOUNDED PRECEDING) s FROM m", NO_ANSI),
    ("long_overflow", "SELECT i, sum(x) OVER (ORDER BY i ROWS UNBOUNDED PRECEDING) s FROM l", {}),
    (
        "long_overflow_noansi",
        "SELECT i, sum(x) OVER (ORDER BY i ROWS UNBOUNDED PRECEDING) s, sum(x) OVER () w FROM l",
        NO_ANSI,
    ),
    ("count_distinct", "SELECT count(DISTINCT v) OVER () FROM t", {}),
    # ---- several windows, expressions over windows, lateral aliases ----
    (
        "expressions",
        "SELECT row_number() OVER (ORDER BY o, p, v) + 1, -rank() OVER (ORDER BY o),"
        " CAST(sum(v) OVER () AS STRING) FROM t",
        {},
    ),
    (
        "several_specs",
        "SELECT p, sum(v) OVER (PARTITION BY p) a, rank() OVER (ORDER BY v) b,"
        " count(*) OVER (PARTITION BY p ORDER BY o) c, max(o) OVER () d FROM t",
        {},
    ),
    ("nested_window", "SELECT rank() OVER (ORDER BY sum(v) OVER ()) FROM t", {}),
    ("window_in_window_args", "SELECT sum(rank() OVER (ORDER BY o)) OVER () FROM t", {}),
    ("lateral_alias_of_window", "SELECT rank() OVER (ORDER BY o) AS r, r + 1 FROM t", {}),
    ("lateral_alias_outside_window", "SELECT o AS x, rank() OVER (ORDER BY o) + x FROM t", {}),
    ("lateral_alias_in_window", "SELECT p AS q, sum(v) OVER (PARTITION BY q) FROM t", {}),
    ("distinct_after_window", "SELECT DISTINCT p, rank() OVER (ORDER BY p) FROM t", {}),
    (
        "order_by_window_alias",
        "SELECT p, row_number() OVER (PARTITION BY p ORDER BY o, v) AS rn FROM t ORDER BY rn, p",
        {},
    ),
    ("order_by_ordinal", "SELECT p, rank() OVER (PARTITION BY p ORDER BY o) FROM t ORDER BY 2, 1", {}),
    ("where_below_window", "SELECT p, sum(v) OVER (PARTITION BY p ORDER BY o) FROM t WHERE v > 1", {}),
    (
        "top_n_per_group",
        "SELECT p, v FROM (SELECT p, v, rank() OVER (PARTITION BY p ORDER BY v DESC) r FROM t) WHERE r <= 2",
        {},
    ),
    ("in_cte", "WITH w AS (SELECT p, v, sum(v) OVER (PARTITION BY p) s FROM t) SELECT p, v, s, v / s FROM w", {}),
    (
        "distribute_sort_cluster",
        "SELECT sum(v) OVER (SORT BY o) a, sum(v) OVER (DISTRIBUTE BY p SORT BY o) b,"
        " sum(v) OVER (CLUSTER BY p) c FROM t",
        {},
    ),
    # ---- named windows ----
    ("named", "SELECT sum(v) OVER w FROM t WINDOW w AS (PARTITION BY p ORDER BY o)", {}),
    ("named_parenthesized", "SELECT sum(v) OVER (w) FROM t WINDOW w AS (PARTITION BY p)", {}),
    (
        "named_two",
        "SELECT sum(v) OVER w, rank() OVER w2 FROM t WINDOW w AS (PARTITION BY p), w2 AS (PARTITION BY p ORDER BY o)",
        {},
    ),
    ("named_alias", "SELECT sum(v) OVER w FROM t WINDOW w AS w2, w2 AS (PARTITION BY p)", {}),
    ("named_derived_table", "SELECT s FROM (SELECT sum(v) OVER w AS s FROM t) WINDOW w AS (PARTITION BY p)", {}),
    ("named_missing", "SELECT sum(v) OVER w FROM t WINDOW x AS (PARTITION BY p)", {}),
    ("named_no_clause", "SELECT sum(v) OVER w FROM t", {}),
    ("named_missing_before_table", "SELECT sum(v) OVER w FROM nosuch WINDOW x AS (PARTITION BY p)", {}),
    ("named_no_clause_table", "SELECT sum(v) OVER w FROM nosuch", {}),
    ("named_in_subquery", "SELECT (SELECT max(v) OVER w FROM t LIMIT 1) FROM t WINDOW w AS (PARTITION BY p)", {}),
    ("named_unresolved_reference", "SELECT sum(v) OVER w FROM t WINDOW w AS w2", {}),
    ("named_invalid_reference", "SELECT sum(v) OVER w FROM t WINDOW w AS w2, w2 AS w3, w3 AS (PARTITION BY p)", {}),
    ("named_repetitive", "SELECT sum(v) OVER w FROM t WINDOW w AS (PARTITION BY p), w AS (ORDER BY o)", {}),
    ("named_with_group_by", "SELECT sum(v) OVER w FROM t GROUP BY v WINDOW w AS (PARTITION BY v)", {}),
    # ---- windows over aggregates ----
    (
        "over_aggregate",
        "SELECT p, sum(sum(v)) OVER (PARTITION BY p), rank() OVER (ORDER BY sum(v)) FROM t GROUP BY p",
        {},
    ),
    (
        "over_aggregate_having",
        "SELECT p, sum(v) AS s, rank() OVER (ORDER BY sum(v)) FROM t GROUP BY p HAVING s > 3",
        {},
    ),
    (
        "over_aggregate_having_window_alias",
        "SELECT p, sum(v) OVER (PARTITION BY p) AS s FROM t GROUP BY p, v HAVING s > 3",
        {},
    ),
    ("over_aggregate_order", "SELECT p, max(o), rank() OVER (PARTITION BY p ORDER BY max(o)) FROM t GROUP BY p", {}),
    ("over_aggregate_missing", "SELECT p, sum(v) OVER (PARTITION BY o) FROM t GROUP BY p", {}),
    ("over_aggregate_missing_order", "SELECT p, sum(v) OVER (ORDER BY max(o)) FROM t GROUP BY p", {}),
    (
        "over_aggregate_order_by_input",
        "SELECT p, max(v) OVER (PARTITION BY p) FROM t GROUP BY p, v ORDER BY sum(o)",
        {},
    ),
    ("over_global_aggregate", "SELECT sum(v), rank() OVER (ORDER BY sum(v)) FROM t", {}),
    ("over_global_aggregate_count", "SELECT count(*) OVER (), count(*) FROM t", {}),
    ("over_global_having", "SELECT max(v) OVER () FROM t HAVING count(*) > 1", {}),
    (
        "over_aggregate_lca_having",
        "SELECT p AS q, q + 1 AS r, rank() OVER (ORDER BY p) FROM t GROUP BY p HAVING count(*) > 0",
        {},
    ),
    ("having_no_group_by", "SELECT v FROM t HAVING count(*) > 1", {}),
    ("order_by_grouping_key_through_window", "SELECT p, rank() OVER (ORDER BY p) FROM t GROUP BY p, v ORDER BY v", {}),
    ("order_by_input_through_window", "SELECT p, rank() OVER (ORDER BY p) FROM t ORDER BY v", {}),
    ("order_by_ungrouped_through_window", "SELECT p, rank() OVER (ORDER BY p) FROM t GROUP BY p ORDER BY o", {}),
    (
        "over_aggregate_group_by_expr",
        "SELECT p + 1 AS k, sum(v), sum(sum(v)) OVER (ORDER BY p + 1) FROM t GROUP BY p + 1",
        {},
    ),
    # ---- errors ----
    ("where_window", "SELECT * FROM t WHERE rank() OVER (ORDER BY o) = 1", {}),
    ("having_window", "SELECT p, count(*) FROM t GROUP BY p HAVING rank() OVER (ORDER BY p) = 1", {}),
    ("order_by_window", "SELECT p FROM t ORDER BY rank() OVER (ORDER BY o)", {}),
    ("group_by_window", "SELECT count(*) FROM t GROUP BY rank() OVER (ORDER BY o)", {}),
    ("join_window", "SELECT * FROM t a JOIN t b ON rank() OVER (ORDER BY a.o) = b.o", {}),
    ("window_in_aggregate", "SELECT sum(rank() OVER (ORDER BY o)) FROM t", {}),
    ("window_in_filter", "SELECT sum(v) FILTER (WHERE rank() OVER (ORDER BY o) > 1) FROM t", {}),
    ("distinct_window", "SELECT sum(DISTINCT v) OVER (PARTITION BY p) FROM t", {}),
    ("scalar_over", "SELECT upper('a') OVER () FROM t", {}),
    ("without_over", "SELECT rank() FROM t", {}),
    ("without_over_cume_dist", "SELECT cume_dist() FROM t", {}),
    ("without_over_ntile", "SELECT ntile(2) FROM t", {}),
    ("not_ordered", "SELECT row_number() OVER (PARTITION BY p) FROM t", {}),
    ("not_ordered_rank", "SELECT rank() OVER (PARTITION BY p) FROM t", {}),
    ("not_ordered_bad_frame", "SELECT row_number() OVER (ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t", {}),
    (
        "frame_mismatch_rows",
        "SELECT rank() OVER (ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM t",
        {},
    ),
    (
        "frame_mismatch_range",
        "SELECT row_number() OVER (ORDER BY o RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        {},
    ),
    (
        "frame_mismatch_cume_dist",
        "SELECT cume_dist() OVER (ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        {},
    ),
    ("frame_mismatch_offsets", "SELECT rank() OVER (ORDER BY o ROWS BETWEEN 2 PRECEDING AND 3 FOLLOWING) FROM t", {}),
    (
        "frame_match",
        "SELECT row_number() OVER (ORDER BY o, p, v ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),"
        " cume_dist() OVER (ORDER BY o RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        {},
    ),
    (
        "invalid_bound_order",
        "SELECT sum(v) OVER (ORDER BY o ROWS BETWEEN UNBOUNDED FOLLOWING AND CURRENT ROW) FROM t",
        {},
    ),
    (
        "invalid_bound_order_2",
        "SELECT sum(v) OVER (ORDER BY o ROWS BETWEEN CURRENT ROW AND UNBOUNDED PRECEDING) FROM t",
        {},
    ),
    (
        "range_without_order",
        "SELECT sum(v) OVER (PARTITION BY p RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        {},
    ),
    (
        "range_without_order_unbounded",
        "SELECT sum(v) OVER (PARTITION BY p RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM t",
        {},
    ),
    ("ntile_zero", "SELECT ntile(0) OVER (ORDER BY o) FROM t", {}),
    ("ntile_negative", "SELECT ntile(-3) OVER (ORDER BY o) FROM t", {}),
    ("ntile_column", "SELECT ntile(o) OVER (ORDER BY o) FROM t", {}),
    ("ntile_bigint", "SELECT ntile(2L) OVER (ORDER BY o) FROM t", {}),
    ("ntile_string", "SELECT ntile('2') OVER (ORDER BY o) FROM t", {}),
    ("ntile_null", "SELECT ntile(NULL) OVER (ORDER BY o) FROM t", {}),
    # ---- later parts of P2-T055: value-bound frames ----
    ("rows_sliding", "SELECT o, v, sum(v) OVER (ORDER BY o, p, v ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t", {}),
    ("range_offset", "SELECT o, sum(v) OVER (ORDER BY o RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t", {}),
    (
        "rows_current_to_end",
        "SELECT o, v, sum(v) OVER (ORDER BY o, p, v ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM t",
        {},
    ),
]


def view_sql(name: str, columns: str, rows: list[tuple[str | None, ...]]) -> str:
    cols = [c.strip().split(" ", 1) for c in columns.split(", ")]

    def lit(text: str | None, ty: str) -> str:
        if text is None:
            return f"CAST(NULL AS {ty})"
        quoted = text.replace("'", "''")
        return f"CAST('{quoted}' AS {ty})"

    values = ", ".join("(" + ", ".join(lit(v, ty) for v, (_, ty) in zip(r, cols, strict=True)) + ")" for r in rows)
    names = ", ".join(n for n, _ in cols)
    return f"CREATE OR REPLACE TEMPORARY VIEW {name} AS SELECT * FROM VALUES {values} AS {name}({names})"


def text_of(value, type_name: str) -> str | None:
    """A result value as the fixture records it."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else repr(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, datetime.date):
        return value.isoformat()
    raise SystemExit(f"unexpected {type(value).__name__} value for {type_name}")


def record_error(e) -> dict:
    cond = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", None)
    params = e.getMessageParameters() if hasattr(e, "getMessageParameters") else {}
    return {
        "error_class": cond() if cond else "UNCLASSIFIED",
        "sql_state": e.getSqlState() if hasattr(e, "getSqlState") else None,
        "message_parameters": dict(sorted((params or {}).items())),
    }


def main() -> None:
    from pyspark.errors import PySparkException
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://127.0.0.1:15012")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/window-goldens.json")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    if spark.version != "4.2.0":
        raise SystemExit(f"expected Spark 4.2.0, found {spark.version}")
    tables = {}
    for name, (columns, rows) in TABLES.items():
        spark.sql(view_sql(name, columns, rows))
        schema = spark.table(name).schema
        tables[name] = {
            "columns": [
                {"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in schema.fields
            ],
            "rows": [list(r) for r in rows],
        }

    cases = []
    for case_id, sql, conf in CASES:
        for k, v in conf.items():
            spark.conf.set(k, v)
        try:
            df = spark.sql(sql)
            fields = df.schema.fields
            rows = df.collect()
            types = [f.dataType.simpleString() for f in fields]
            out = [[text_of(v, ty) for v, ty in zip(r, types, strict=True)] for r in rows]
            result = {
                "ok": True,
                "output": [
                    {"name": f.name, "type": ty, "nullable": f.nullable} for f, ty in zip(fields, types, strict=True)
                ],
                "rows": sorted(out, key=lambda r: json.dumps(r)),
            }
        except PySparkException as e:
            result = {"ok": False, **record_error(e)}
        finally:
            for k in conf:
                spark.conf.unset(k)
        cases.append({"id": case_id, "sql": sql, "conf": conf, **result})

    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": spark.version,
        "generator": GENERATOR,
        "notes": "Output columns and rows (as text, sorted) or error (condition, SQLSTATE, message parameters) "
        "of each case over the temporary views `tables` (rows as the text each value is cast from).",
        "tables": tables,
        "cases": cases,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
