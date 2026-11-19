#!/usr/bin/env python3
"""Generate the Spark 4.3 coverage fixture (P3 PR-05, the 4.3 coverage list of P3-T090).

Upstream's 4.3.0 goldens (`tests/spark-golden/profiles/v4.3.0`) do not exercise everything the 4.3 migration
changed. This fixture covers what they miss, from the list in plan row PR-05: new functions, new syntax, behaviour
changes, and the correctness fixes worth testing on every version. Every expectation comes from a REAL Spark 4.3.0
run (Spark Connect), never from memory: a case records the rows Spark returned (as text), or the error class it
raised, or, for output that holds paths and timestamps, only the column names.

Each case runs in a fresh session with ANSI on. Setup statements create temporary views and, for the DDL cases, a
table in the session catalog of the throw-away server (a private container; the shared oracle takes no tables).
Output is deterministic: rows are sorted unless the query orders them, and no local path reaches the file.

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.3.0' \\
      python tests/scripts/gen_spark43_coverage_goldens.py --remote sc://127.0.0.1:15031 \\
      --out tests/fixtures/spark-4.3.0/spark43-coverage.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_spark43_coverage_goldens.py"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\|/tmp/|spark-warehouse")

KV = "values (1, 10), (2, 20), (3, 30) as t(k, v)"
VIEW_T = f"create temporary view t as select * from {KV}"
TIME_ON = "set spark.sql.timeType.enabled=true"
NANOS_ON = "set spark.sql.timestampNanosTypes.enabled=true"
LEGACY_STORE = "set spark.sql.storeAssignmentPolicy=LEGACY"
TBL = "create table tbl (id int, name string) using parquet"

# (id, area, setup, sql, mode) where mode is "rows" (default) or "schema" (column names only).
CASES: list[tuple[str, str, list[str], str, str]] = [
    # --- functions -------------------------------------------------------------------------------------------
    (
        "fn_collect_union",
        "function",
        [],
        "select collect_union(x) from values (array(1,2)),(array(2,3)) as t(x)",
        "rows",
    ),
    (
        "fn_collect_union_nulls",
        "function",
        [],
        "select collect_union(x) from values (array(1,null)),(null),(array(null,2)) as t(x)",
        "rows",
    ),
    (
        "fn_collect_union_empty",
        "function",
        [],
        "select collect_union(x) from values (array(1)) as t(x) where false",
        "rows",
    ),
    (
        "fn_to_base32",
        "function",
        [],
        "select to_base32(cast('abc' as binary)), to_base32(cast('' as binary)), to_base32(cast(null as binary))",
        "rows",
    ),
    (
        "fn_from_base32",
        "function",
        [],
        "select cast(from_base32('MFRGG===') as string), from_base32(cast(null as string))",
        "rows",
    ),
    ("fn_from_base32_invalid", "function", [], "select from_base32('!!!')", "rows"),
    (
        "fn_variant_insert",
        "function",
        [],
        "select variant_insert(parse_json('{\"a\":1}'), '$.b', parse_json('2'))",
        "rows",
    ),
    (
        "fn_variant_insert_nested_missing",
        "function",
        [],
        "select variant_insert(parse_json('{\"a\":1}'), '$.b.c', parse_json('2'))",
        "rows",
    ),
    (
        "fn_try_variant_insert",
        "function",
        [],
        "select try_variant_insert(parse_json('{\"a\":1}'), '$.b.c', parse_json('2')), "
        "try_variant_insert(parse_json('[1]'), '$.b', parse_json('2'))",
        "rows",
    ),
    (
        "fn_variant_array_append",
        "function",
        [],
        "select variant_array_append(parse_json('[1,2]'), '$', parse_json('3'))",
        "rows",
    ),
    (
        "fn_variant_array_append_nested",
        "function",
        [],
        "select variant_array_append(parse_json('{\"a\":[1]}'), '$.a', parse_json('2'))",
        "rows",
    ),
    (
        "fn_try_variant_array_append",
        "function",
        [],
        "select try_variant_array_append(parse_json('{\"a\":1}'), '$.a', parse_json('3')), "
        "try_variant_array_append(parse_json('[1]'), '$', parse_json('3'))",
        "rows",
    ),
    (
        "fn_variant_strip_nulls",
        "function",
        [],
        'select variant_strip_nulls(parse_json(\'{"a":null,"b":[1,null],"c":{"d":null}}\'))',
        "rows",
    ),
    (
        "fn_variant_strip_nulls_scalar",
        "function",
        [],
        "select variant_strip_nulls(parse_json('null')), variant_strip_nulls(parse_json('1'))",
        "rows",
    ),
    (
        "fn_try_variant_set",
        "function",
        [],
        "select try_variant_set(parse_json('{\"a\":1}'), '$.a', parse_json('5')), "
        "try_variant_set(parse_json('{\"a\":1}'), '$.b.c', parse_json('5'))",
        "rows",
    ),
    ("fn_localtime", "function", [], "select typeof(localtime()), typeof(localtime(3)), typeof(localtime(0))", "rows"),
    ("fn_localtime_precision_error", "function", [], "select localtime(10)", "rows"),
    # TIME and the nanosecond types are off by default: the default is an error, and each feature is also tested on.
    ("fn_time_disabled", "function", [], "select time'12:34:56'", "rows"),
    ("fn_time_precision_9_disabled", "function", [], "select cast(time'12:34:56.123456789' as time(9))", "rows"),
    ("fn_timestamp_nanos_disabled", "function", [], "select current_timestamp(9)", "rows"),
    (
        "fn_time_precision_9",
        "function",
        [TIME_ON],
        "select cast(cast(time'12:34:56.123456789' as time(9)) as string)",
        "rows",
    ),
    (
        "fn_time_precision_6",
        "function",
        [TIME_ON],
        "select cast(cast(time'12:34:56.123456789' as time(6)) as string), cast(cast('10:00:00' as time(3)) as string)",
        "rows",
    ),
    (
        "fn_time_precision_0",
        "function",
        [TIME_ON],
        "select cast(cast(time'12:34:56.999999999' as time(0)) as string)",
        "rows",
    ),
    (
        "fn_time_typeof_precisions",
        "function",
        [TIME_ON],
        "select typeof(cast('10:00:00' as time(9))), typeof(cast('10:00:00' as time(3))), typeof(time'01:02:03')",
        "rows",
    ),
    (
        "fn_timestamp_ntz_to_time",
        "function",
        [TIME_ON],
        "select cast(cast(timestamp_ntz'2024-01-02 03:04:05.123456' as time(6)) as string)",
        "rows",
    ),
    (
        "fn_timestamp_ntz_to_time_9",
        "function",
        [TIME_ON, NANOS_ON],
        "select cast(cast(timestamp_ntz'2024-01-02 03:04:05.123456789' as time(9)) as string)",
        "rows",
    ),
    (
        "fn_time_to_timestamp_ntz_nanos",
        "function",
        [TIME_ON, NANOS_ON],
        "select cast(time'12:34:56.789' as timestamp_ntz(9))",
        "rows",
    ),
    (
        "fn_current_timestamp_precision",
        "function",
        [],
        "select typeof(current_timestamp(3)), typeof(current_timestamp(6))",
        "rows",
    ),
    ("fn_current_timestamp_precision_9", "function", [NANOS_ON], "select typeof(current_timestamp(9))", "rows"),
    ("fn_localtime_9", "function", [TIME_ON], "select typeof(localtime(9))", "rows"),
    # --- behaviour changes ---------------------------------------------------------------------------------------
    ("beh_ceil_double_overflow", "behaviour", [], "select ceil(9223372036854775807.5d)", "rows"),
    ("beh_floor_double_overflow", "behaviour", [], "select floor(cast(-9.3e18 as double))", "rows"),
    ("beh_ceil_decimal_wide", "behaviour", [], "select ceil(cast(9223372036854775807 as decimal(19,0)) + 0.5)", "rows"),
    (
        "beh_ceil_in_range",
        "behaviour",
        [],
        "select ceil(1.2d), floor(-1.2d), ceil(cast(null as double)), ceil(cast('NaN' as double))",
        "rows",
    ),
    (
        "beh_grouping_sets_empty_input",
        "behaviour",
        [],
        "select count(*) from (select 1 as a where false) group by grouping sets (())",
        "rows",
    ),
    (
        "beh_grouping_sets_empty_sum",
        "behaviour",
        [],
        "select sum(a), count(a) from (select 1 as a where false) group by grouping sets (())",
        "rows",
    ),
    (
        "beh_grouping_sets_empty_with_key",
        "behaviour",
        [],
        "select a, count(*) from (select 1 as a where false) group by grouping sets ((a), ())",
        "rows",
    ),
    (
        "beh_rollup_empty_input",
        "behaviour",
        [],
        "select a, count(*) from (select 1 as a where false) group by rollup(a)",
        "rows",
    ),
    ("beh_plain_aggregate_empty", "behaviour", [], "select count(*), sum(a) from (select 1 as a where false)", "rows"),
    ("beh_group_by_empty_parens", "behaviour", [], "select count(*) from (select 1 a where false) group by ()", "rows"),
    # --- correctness fixes, to test on every version ---------------------------------------------------------------
    (
        "fix_in_subquery_null_hit",
        "fix",
        [],
        "select 1 where 1 in (select cast(null as int) union all select 1)",
        "rows",
    ),
    (
        "fix_in_subquery_null_miss",
        "fix",
        [],
        "select count(*) from values (1) as a(x) where 2 in (select cast(null as int) union all select 1)",
        "rows",
    ),
    (
        "fix_in_subquery_null_projected",
        "fix",
        [],
        "select x, x in (select y from values (1),(cast(null as int)) as b(y)) as r from values (1),(2) as a(x)",
        "rows",
    ),
    (
        "fix_not_in_subquery_null",
        "fix",
        [],
        "select x from values (1),(2),(3) as a(x) where x not in (select cast(null as int) union all select 1)",
        "rows",
    ),
    (
        "fix_in_subquery_null_left",
        "fix",
        [],
        "select x from values (1),(cast(null as int)) as a(x) where x in (select y from values (1),(2) as b(y))",
        "rows",
    ),
    (
        "fix_pivot_duplicate_values",
        "fix",
        [],
        "select * from (select * from values ('a',1),('a',2),('b',3) as t(k,v)) pivot (sum(v) for k in ('a','b','a'))",
        "rows",
    ),
    (
        "fix_pivot_duplicate_alias",
        "fix",
        [],
        "select * from (select * from values ('a',1),('b',3) as t(k,v)) "
        "pivot (sum(v) as s for k in ('a' as x,'b' as x))",
        "rows",
    ),
    (
        "fix_cte_nondeterministic_pushdown",
        "fix",
        [],
        "with c as (select id, rand(1) r from range(10)) select count(*) from c where r < 2 and r > -1",
        "rows",
    ),
    (
        "fix_cte_nondeterministic_two_refs",
        "fix",
        [],
        "with c as (select id, rand(3) r from range(6)) select count(*) from c a join c b on a.id = b.id where a.r < 2",
        "rows",
    ),
    (
        "fix_join_null_safe_equality",
        "fix",
        [],
        "select a.x from values (1),(null) as a(x) join values (1),(null) as b(x) on a.x <=> b.x and a.x is not null",
        "rows",
    ),
    (
        "fix_join_null_safe_full",
        "fix",
        [],
        "select a.x, b.x from values (1),(null) as a(x) full join values (1),(null) as b(x) on a.x <=> b.x",
        "rows",
    ),
    (
        "fix_join_or_null",
        "fix",
        [],
        "select count(*) from values (1),(null) as a(x) join values (1),(null) as b(x) on a.x = b.x or a.x is null",
        "rows",
    ),
    (
        "fix_insert_struct_by_position",
        "fix",
        [
            "create table s1 (a struct<x:int,y:string>) using parquet",
            "insert into s1 values (named_struct('x', 1, 'y', 'u'))",
            "create table s2 (a struct<y:string,x:int>) using parquet",
        ],
        "insert into s2 (a) select a from s1",
        "rows",
    ),
    (
        "fix_insert_struct_by_position_legacy",
        "fix",
        [
            LEGACY_STORE,
            "create table s1 (a struct<x:int,y:string>) using parquet",
            "insert into s1 values (named_struct('x', 1, 'y', 'u'))",
            "create table s2 (a struct<y:string,x:int>) using parquet",
            "insert into s2 (a) select a from s1",
        ],
        "select a.x, a.y from s2",
        "rows",
    ),
    (
        "fix_insert_struct_in_array",
        "fix",
        [
            "create table a1 (v array<struct<x:int,y:string>>) using parquet",
            "insert into a1 values (array(named_struct('x', 1, 'y', 'u')))",
            "create table a2 (v array<struct<y:string,x:int>>) using parquet",
        ],
        "insert into a2 (v) select v from a1",
        "rows",
    ),
    (
        "fix_insert_struct_in_map",
        "fix",
        [
            "create table m1 (v map<string,struct<x:int,y:string>>) using parquet",
            "insert into m1 values (map('k', named_struct('x', 1, 'y', 'u')))",
            "create table m2 (v map<string,struct<y:string,x:int>>) using parquet",
        ],
        "insert into m2 (v) select v from m1",
        "rows",
    ),
    # --- syntax ------------------------------------------------------------------------------------------------
    ("syn_rebalance_clause", "syntax", [], "select * from range(3) rebalance", "rows"),
    ("syn_rebalance_hint", "syntax", [], "select /*+ REBALANCE(2) */ * from range(3)", "rows"),
    ("syn_rebalance_advisory", "syntax", [], "select /*+ REBALANCE(2, id) */ * from range(3)", "rows"),
    ("syn_comment_on_column", "syntax", [TBL], "comment on column tbl.name is 'the name'", "rows"),
    (
        "syn_comment_on_column_describe",
        "syntax",
        [TBL, "comment on column tbl.name is 'the name'"],
        "describe tbl",
        "rows",
    ),
    ("syn_comment_on_column_missing", "syntax", [TBL], "comment on column tbl.nope is 'x'", "rows"),
    ("syn_show_tables_as_json", "syntax", [TBL], "show tables as json", "schema"),
    ("syn_show_table_extended_as_json", "syntax", [TBL], "show table extended like 'tbl' as json", "schema"),
    (
        "syn_show_partitions_as_json",
        "syntax",
        ["create table pt (id int, p int) using parquet partitioned by (p)", "insert into pt values (1, 7)"],
        "show partitions pt as json",
        "schema",
    ),
    ("syn_show_columns_plain", "syntax", [TBL], "show columns in tbl", "rows"),
    ("syn_time_travel_at_version", "syntax", [TBL], "select * from tbl@v1", "rows"),
    ("syn_time_travel_at_timestamp", "syntax", [TBL], "select * from tbl@20240101000000000", "rows"),
    ("syn_time_travel_clause_and_at", "syntax", [TBL], "select * from tbl@v1 version as of 1", "rows"),
    ("syn_insert_replace_where", "syntax", [TBL], "insert into tbl replace where id = 1 values (1, 'a')", "rows"),
    (
        "syn_insert_replace_where_columns",
        "syntax",
        [TBL],
        "insert into tbl (id, name) replace where id = 1 values (1, 'a')",
        "rows",
    ),
    ("syn_update_with_options", "syntax", [TBL], "update tbl with ('k' = 'v') set name = 'x' where id = 1", "rows"),
    ("syn_delete_with_options", "syntax", [TBL], "delete from tbl with ('k' = 'v') where id = 1", "rows"),
    (
        "syn_merge_with_options",
        "syntax",
        [TBL, "create temporary view src as select 1 as id, 'z' as name"],
        "merge into tbl with ('k' = 'v') using src on tbl.id = src.id when matched then update set name = src.name",
        "rows",
    ),
    ("syn_bin_by", "syntax", [VIEW_T], "select k, count(*) from t bin by k width 10", "rows"),
    ("syn_scalar_udf_in_tvf_argument", "syntax", [], "select * from range(1, 4, 1, abs(-2))", "rows"),
    ("syn_scalar_subquery_in_tvf_argument", "syntax", [], "select * from range((select 2))", "rows"),
    (
        "syn_scalar_function_in_tvf_argument",
        "syntax",
        [],
        "select * from explode(array(abs(-1), length('ab')))",
        "rows",
    ),
]


def error_class(e: Exception) -> str:
    m = re.search(r"\[([A-Z_.]+)\]", str(e))
    return m.group(1) if m else "UNKNOWN"


def outcome(spark, sql: str, mode: str, ordered: bool) -> dict:
    """The rows Spark returned (every cell as Spark's own `cast(c as string)`), the column names, or the error class."""
    try:
        df = spark.sql(sql)
        fields = df.schema.fields
        if mode == "schema":
            df.collect()
            return {"columns": [f.name for f in fields]}
        if not fields:  # a command: nothing comes back
            df.collect()
            return {"rows": [], "ncols": 0}
        names = ", ".join(f"c{i}" for i in range(len(fields)))
        casts = ", ".join(f"cast(c{i} as string)" for i in range(len(fields)))
        wrapped = True
        try:
            collected = spark.sql(f"select {casts} from ({sql}) as q({names})").collect()
        except Exception:  # a command such as DESCRIBE cannot be a subquery: its rows are text already
            wrapped = False
            collected = df.collect()
        rows = [[None if v is None else str(v) for v in r] for r in collected]
        return {
            "rows": rows if ordered else sorted(rows, key=lambda r: [x or "" for x in r]),
            "ncols": len(fields),
            "wrapped": wrapped,
        }
    except Exception as e:  # an expected outcome of a case: the error class
        return {"error_class": error_class(e)}


def run(remote: str) -> list[dict]:
    from pyspark.sql import SparkSession

    out = []
    for cid, area, setup, sql, mode in CASES:
        spark = SparkSession.builder.remote(remote).create()
        try:
            spark.sql("SET spark.sql.ansi.enabled=true").collect()
            failed_setup = None
            for stmt in setup:
                try:
                    spark.sql(stmt).collect()
                except Exception as e:
                    failed_setup = error_class(e)
                    break
            case = {"id": cid, "area": area, "setup": setup, "sql": sql, "mode": mode}
            if failed_setup:
                case["setup_error_class"] = failed_setup
            else:
                case.update(outcome(spark, sql, mode, " order by " in sql.lower()))
            out.append(case)
        finally:
            for t in ("tbl", "pt", "s1", "s2", "a1", "a2", "m1", "m2"):
                with contextlib.suppress(Exception):
                    spark.sql(f"drop table if exists {t}").collect()
            spark.stop()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pyspark.sql import SparkSession

    s = SparkSession.builder.remote(a.remote).create()
    version = s.version
    s.stop()
    cases = run(a.remote)
    doc = {"format_version": FORMAT_VERSION, "generator": GENERATOR, "spark_version": version, "cases": cases}
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    assert not LOCAL_PATH.search(text), "local path in fixture"
    Path(a.out).write_text(text)
    by_kind: dict[str, int] = {}
    for c in cases:
        k = "rows" if "rows" in c else "columns" if "columns" in c else "error"
        by_kind[k] = by_kind.get(k, 0) + 1
    print("wrote", a.out, len(cases), "cases, Spark", version, by_kind)


if __name__ == "__main__":
    main()
