#!/usr/bin/env python3
"""Generate the `concat_ws` with ARRAY arguments golden fixture (P3-1, review findings B1, B2, S1).

Every expectation comes from a REAL Spark run over Spark Connect, never from docs, source or memory.
Each case is a literal-only query (no tables), run with ANSI on (the 4.x default) and with
`spark.sql.ansi.enabled=false`. The fixture records the output column (name, `simpleString` type,
nullability) and the single row as text (NULL as `null`), or the error condition, SQLSTATE, message
parameters and the rendered message (which pins the `requiredType` text of the second parameter,
`("ARRAY<STRING>" or "STRING")`). Output is deterministic and contains no local paths.

Usage (from repo root), against a Spark Connect server (the fixture records its version):

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \\
      python tests/scripts/gen_concat_ws_array_goldens.py --remote sc://127.0.0.1:15092 \\
      --out tests/fixtures/spark-4.2.0/concat-ws-array-goldens.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_concat_ws_array_goldens.py"
LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")
MODES = {"ansi": "true", "legacy": "false"}

# (id, expression); the query is `SELECT <expression>`.
CASES: list[tuple[str, str]] = [
    # ---- flatten, NULL elements, NULL arrays, empty arrays ----
    ("flatten", "concat_ws(',', array('a', 'b'), array('c'))"),
    ("flatten_mixed", "concat_ws(',', 'x', array('a', 'b'), 'y', array('c'))"),
    ("null_element_skipped", "concat_ws(',', array('a', NULL, 'b'))"),
    ("all_null_elements", "concat_ws(',', array(CAST(NULL AS STRING), CAST(NULL AS STRING)))"),
    ("null_array_skipped", "concat_ws(',', CAST(NULL AS ARRAY<STRING>), 'z')"),
    ("only_null_array", "concat_ws(',', CAST(NULL AS ARRAY<STRING>))"),
    ("untyped_null_with_array", "concat_ws(',', NULL, array('a'))"),
    ("empty_array", "concat_ws(',', array())"),
    ("empty_array_typed", "concat_ws(',', CAST(array() AS ARRAY<STRING>))"),
    ("empty_array_with_string", "concat_ws(',', 'a', array(), 'b')"),
    ("mixed_all", "concat_ws(',', array('a', NULL, 'b'), 'z', NULL, array())"),
    ("empty_separator", "concat_ws('', array('a', 'b'), 'c')"),
    ("multichar_separator", "concat_ws('--', array('a', 'b'), array('c'))"),
    ("empty_string_elements", "concat_ws(',', array('', 'a', ''))"),
    ("array_only_one_element", "concat_ws('|', array('only'))"),
    ("string_only_unchanged", "concat_ws(',', 'a', NULL, 'b')"),
    ("no_arguments", "concat_ws(',')"),
    ("unicode", "concat_ws('→', array('é', '日本'), 'z')"),
    # ---- NULL separator ----
    ("null_separator", "concat_ws(NULL, array('a', 'b'))"),
    ("null_separator_typed", "concat_ws(CAST(NULL AS STRING), array('a'), 'b')"),
    ("null_separator_empty_array", "concat_ws(NULL, array())"),
    # ---- atomic element types are cast to string ----
    ("array_int", "concat_ws(',', array(1, 2, 3))"),
    ("array_int_null", "concat_ws(',', array(1, NULL, 3))"),
    ("array_bigint", "concat_ws(',', array(9223372036854775807L, -1L))"),
    ("array_double", "concat_ws(',', array(1.5D, CAST('NaN' AS DOUBLE), CAST('-Infinity' AS DOUBLE), -0.0D))"),
    ("array_decimal", "concat_ws(',', array(CAST(1.50 AS DECIMAL(5,2)), CAST(-0.10 AS DECIMAL(5,2))))"),
    ("array_boolean", "concat_ws(',', array(true, false))"),
    ("array_boolean_null", "concat_ws(',', array(true, NULL))"),
    ("array_date", "concat_ws(',', array(DATE'2024-02-29', DATE'1970-01-01'))"),
    ("array_timestamp_ntz", "concat_ws(',', array(TIMESTAMP_NTZ'2024-02-29 12:34:56.789'))"),
    ("array_binary", "concat_ws(',', array(X'41', X'4243'))"),
    ("array_binary_non_utf8", "concat_ws(',', array(X'FF'))"),
    ("array_binary_empty", "concat_ws(',', array(X''))"),
    ("array_interval_day_time", "concat_ws(',', array(INTERVAL '1 02:03:04' DAY TO SECOND))"),
    ("array_interval_year_month", "concat_ws(',', array(INTERVAL '1-2' YEAR TO MONTH))"),
    ("array_int_and_string", "concat_ws(',', array(1, 2), array('a'), 'b')"),
    ("array_mixed_element_types", "concat_ws(',', array(1, 'a', true))"),
    ("cast_array_int_to_string_array", "concat_ws(',', CAST(array(1, 2) AS ARRAY<STRING>))"),
    ("non_string_arg_int", "concat_ws(',', 1, array('a'))"),
    ("non_string_arg_date", "concat_ws(',', DATE'2024-02-29', array('a'))"),
    ("non_string_arg_binary", "concat_ws(',', X'41', 'b')"),
    # ---- nested element types are refused (B1) ----
    ("refuse_array_of_array", "concat_ws(',', array(array('a')))"),
    ("refuse_array_of_array_int", "concat_ws(',', array(array(1, 2), array(3)))"),
    ("refuse_array_of_map", "concat_ws(',', array(map('a', 'b')))"),
    ("refuse_array_of_struct", "concat_ws(',', array(named_struct('a', 1)))"),
    ("refuse_array_of_array_empty", "concat_ws(',', array(array()))"),
    ("refuse_array_of_array_typed_null", "concat_ws(',', CAST(NULL AS ARRAY<ARRAY<STRING>>))"),
    ("refuse_array_of_map_second", "concat_ws(',', 'a', array(map('a', 'b')))"),
    ("refuse_array_of_struct_typed", "concat_ws(',', CAST(array(named_struct('a', 'x')) AS ARRAY<STRUCT<a: STRING>>))"),
    # ---- other refused argument types (S1: the requiredType text) ----
    ("refuse_map_argument", "concat_ws(',', map('a', 'b'))"),
    ("refuse_struct_argument", "concat_ws(',', named_struct('a', 1))"),
    ("refuse_array_separator", "concat_ws(array('a'), 'b')"),
    ("refuse_array_separator_only", "concat_ws(array(','), array('a'))"),
    ("refuse_map_second", "concat_ws(',', 'a', map('a', 'b'))"),
    ("refuse_map_third", "concat_ws(',', 'a', 'b', map('a', 'b'))"),
    ("refuse_struct_separator", "concat_ws(named_struct('a', 1), 'b')"),
    ("refuse_map_separator", "concat_ws(map('a', 'b'), 'b')"),
]


def record_error(e) -> dict:
    cond = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", None)
    params = e.getMessageParameters() if hasattr(e, "getMessageParameters") else {}
    # The first line only: the rest is the analyzed plan and a JVM stack trace.
    message = (e.getMessage() if hasattr(e, "getMessage") else str(e)).split("\n", 1)[0]
    return {
        "error_class": cond() if cond else "UNCLASSIFIED",
        "sql_state": e.getSqlState() if hasattr(e, "getSqlState") else None,
        "message_parameters": dict(sorted((params or {}).items())),
        "message": message,
    }


def main() -> None:
    from pyspark.errors import PySparkException
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://127.0.0.1:15092")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/concat-ws-array-goldens.json")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    cases = []
    for case_id, expr in CASES:
        sql = f"SELECT {expr} AS r"
        for mode, ansi in MODES.items():
            spark.conf.set("spark.sql.ansi.enabled", ansi)
            try:
                df = spark.sql(sql)
                f = df.schema.fields[0]
                # The value is read as the hex of its UTF-8 bytes: a result may hold non-UTF-8 bytes (a BINARY
                # element cast to STRING keeps them), which Arrow cannot decode as text.
                hexed = spark.sql(f"SELECT hex(CAST({expr} AS BINARY)) AS h").collect()
                result = {
                    "ok": True,
                    "type": f.dataType.simpleString(),
                    "nullable": f.nullable,
                    "rows_hex": [[r[0]] for r in hexed],
                }
            except PySparkException as e:
                result = {"ok": False, **record_error(e)}
            finally:
                spark.conf.unset("spark.sql.ansi.enabled")
            cases.append({"id": f"{case_id}.{mode}", "sql": sql, "ansi": ansi == "true", **result})

    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": spark.version,
        "generator": GENERATOR,
        "notes": "Each case is a literal-only `SELECT concat_ws(...) AS r` run with ANSI on and off. Result: the "
        "output column's simpleString type and nullability and its row(s) as the hex of the string's bytes "
        "(NULL as null), or the error condition, SQLSTATE, message parameters and rendered message.",
        "cases": cases,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(cases)} cases (Spark {spark.version}) -> {args.out}")


if __name__ == "__main__":
    main()
