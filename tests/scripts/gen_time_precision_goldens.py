#!/usr/bin/env python3
"""Generate the TIME(p <= 9) / LOCALTIME / nanosecond-timestamp goldens (P3 R3, 4.3-D for 0.3).

Every expectation comes from a REAL Spark run in local mode (classic PySpark, not Connect), one run per
release: Spark 4.2.0 and Spark 4.3.0 give one fixture each, and the test runs every case on the matching
Krishiv profile. A case is a SQL statement with optional session configs. For a statement that runs the
fixture records the output column names and every value cast to STRING on the server (`CAST(c AS STRING)`
of each output column), so TIME values keep their nanoseconds (Python rows would hold microseconds); a case
marked `clock` reads a clock: it records the names, and as `typed` the first row's values of its `typeof(..)`
columns (by position), which do not depend on the clock. For a statement that fails it records the error
condition and the message's first line without the `SQLSTATE` suffix and query context.

Usage (from the repository root, Java 17+ on PATH):

    export SPARK_HOME=/path/to/spark-4.3.0-bin-hadoop3
    PY4J=$(ls "$SPARK_HOME"/python/lib/py4j-*.zip)
    PYTHONPATH="$SPARK_HOME/python:$PY4J" \
      python3 tests/scripts/gen_time_precision_goldens.py --out tests/fixtures/spark-4.3.0/time-precision-goldens.json
    <python with pyspark==4.2.0> tests/scripts/gen_time_precision_goldens.py \
      --out tests/fixtures/spark-4.2.0/time-precision-goldens.json

`--remote sc://host:port` runs the cases on a Spark Connect server of the same release instead (a client
of that release, e.g. `pyspark-client==4.3.0`); the values are cast to STRING on the server either way.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_time_precision_goldens.py"
ON = {"spark.sql.timeType.enabled": "true"}
NTZ = {"spark.sql.timestampType": "TIMESTAMP_NTZ"}
NANOS = {"spark.sql.timestampNanosTypes.enabled": "true"}

# (id, configs, sql, clock): `clock` cases read a clock, so only their names are recorded.
CASES: list[tuple[str, dict[str, str], str, bool]] = [
    # TIME(p) type syntax and precision bounds.
    ("time9_null", ON, "SELECT CAST(NULL AS TIME(9)) AS c", False),
    ("time7_null_key_off", {}, "SELECT CAST(NULL AS TIME(7)) AS c", False),
    ("time10", ON, "SELECT CAST(NULL AS TIME(10))", False),
    ("time10_key_off", {}, "SELECT CAST(NULL AS TIME(10))", False),
    ("time_huge", ON, "SELECT CAST(NULL AS TIME(99999999999))", False),
    ("cast_str_time9", ON, "SELECT CAST(CAST('12:00:00.123456789' AS TIME(9)) AS STRING) AS c", False),
    ("cast_str_time7", ON, "SELECT CAST(CAST('12:00:00.123456789' AS TIME(7)) AS STRING) AS c", False),
    ("cast_str_time6", ON, "SELECT CAST(CAST('12:00:00.123456789' AS TIME(6)) AS STRING) AS c", False),
    ("cast_str_time", ON, "SELECT CAST(CAST('12:00:00.123456789' AS TIME) AS STRING) AS c", False),
    ("cast_time9_time3", ON, "SELECT CAST(CAST(TIME'12:00:00.123456789' AS TIME(3)) AS STRING) AS c", False),
    ("cast_time9_time6", ON, "SELECT CAST(CAST(TIME'12:00:00.123456789' AS TIME(6)) AS STRING) AS c", False),
    # Implicit and nested string casts to TIME(3) (lane review round 3, thread on cast.rs).
    (
        "coalesce_time3_str",
        ON,
        "SELECT typeof(c), CAST(c AS STRING) FROM "
        "(SELECT coalesce(CAST(NULL AS TIME(3)), '12:00:00.123456789') AS c)",
        False,
    ),
    ("eq_time3_str", ON, "SELECT CAST(TIME'12:00:00.123' AS TIME(3)) = '12:00:00.1234' AS c", False),
    ("in_time3_str", ON, "SELECT CAST(TIME'12:00:00.123' AS TIME(3)) IN ('12:00:00.1234', '01:00:00') AS c", False),
    (
        "case_time3_str",
        ON,
        "SELECT CAST(CASE WHEN rand() >= 0 THEN '12:00:00.123456789' ELSE CAST(NULL AS TIME(3)) END AS STRING) AS c",
        False,
    ),
    (
        "nested_array_time3_str",
        ON,
        "SELECT CAST(CAST(array('12:00:00.123456789') AS array<time(3)>)[0] AS STRING) AS c",
        False,
    ),
    # Literals: type, name and value.
    (
        "lit_types",
        {},
        "SELECT typeof(TIME'12:00:00'), typeof(TIME'12:00:00.1'), "
        "typeof(TIME'12:00:00.1234567'), typeof(TIME'12:00:00.123456789')",
        False,
    ),
    ("lit_name9", {}, "SELECT TIME'12:00:00.123456789'", False),
    (
        "lit_trailing_zeros",
        {},
        "SELECT CAST(TIME'12:00:00.123456700' AS STRING) AS c, typeof(TIME'12:00:00.123456700') AS t",
        False,
    ),
    ("lit_to_string", {}, "SELECT CAST(TIME'12:00:00.123456789' AS STRING) AS c", False),
    ("lit_10_digits", ON, "SELECT TIME'12:00:00.1234567890'", False),
    # Literals with the TIME key on: the padded name of p > 6 digits, unaliased.
    ("lit_name9_on", ON, "SELECT TIME'12:00:00.123456789'", False),
    ("lit_trailing_zeros_on", ON, "SELECT TIME'12:00:00.123456700', typeof(TIME'12:00:00.123456700')", False),
    ("lit_two_digits_on", ON, "SELECT TIME'12:00:00.10', typeof(TIME'12:00:00.10')", False),
    ("lit_seven_digits_on", ON, "SELECT TIME'12:00:00.1234567', typeof(TIME'12:00:00.1234567')", False),
    ("lit_second_key_off", {}, "SELECT second(TIME'12:00:00.123456789')", False),
    # current_time / LOCALTIME.
    ("current_time9", ON, "SELECT current_time(9), typeof(current_time(9)), typeof(current_time(7))", True),
    ("current_time10", ON, "SELECT current_time(10)", False),
    ("localtime_bare", ON, "SELECT localtime, typeof(localtime)", True),
    ("localtime_call", ON, "SELECT localtime(), localtime(3), typeof(localtime(9))", True),
    ("localtime_key_off", {}, "SELECT localtime(9)", True),
    ("localtime_negative", ON, "SELECT localtime(-1)", False),
    ("localtime_10", ON, "SELECT localtime(10)", False),
    # `currentLike` with a column list ending the item is a multi-name alias (4.3: LOCALTIME too).
    ("localtime_multi_alias", ON, "SELECT localtime(n) FROM VALUES (6) t(n)", False),
    ("current_date_multi_alias", ON, "SELECT current_date(n) FROM VALUES (6) t(n)", False),
    # Expressions over TIME(p > 6).
    (
        "compare",
        ON,
        "SELECT TIME'12:00:00.123456789' = TIME'12:00:00.123456', TIME'12:00:00.123456789' > TIME'12:00:00.123456'",
        False,
    ),
    (
        "common_types",
        ON,
        "SELECT typeof(coalesce(TIME'12:00:00.1', TIME'12:00:00.123456789')), "
        "typeof(array(TIME'12:00:00.1', TIME'12:00:00.123456789')), "
        "typeof(CASE WHEN true THEN TIME'12:00:00.1' ELSE CAST(NULL AS TIME(8)) END)",
        False,
    ),
    (
        "max_distinct",
        ON,
        "SELECT CAST(max(c) AS STRING), typeof(max(c)), count(DISTINCT c) "
        "FROM VALUES (TIME'01:00:00.000000001'), (TIME'01:00:00.000000002') t(c)",
        False,
    ),
    # The same two values apart: `max` (no TIME accumulator yet) and `count(DISTINCT)`.
    (
        "max_only",
        ON,
        "SELECT CAST(max(c) AS STRING), typeof(max(c)) "
        "FROM VALUES (TIME'01:00:00.000000001'), (TIME'01:00:00.000000002') t(c)",
        False,
    ),
    (
        "count_distinct_only",
        ON,
        "SELECT count(DISTINCT c) FROM VALUES (TIME'01:00:00.000000001'), (TIME'01:00:00.000000002') t(c)",
        False,
    ),
    ("hash", ON, "SELECT hash(TIME'12:00:00.123456789'), hash(TIME'12:00:00.123456')", False),
    (
        "extract_second",
        ON,
        "SELECT extract(SECOND FROM TIME'12:00:01.123456789'), "
        "typeof(extract(SECOND FROM CAST('12:00:01.1234567' AS TIME(7)))), "
        "typeof(extract(SECOND FROM TIME'12:00:01.123'))",
        False,
    ),
    ("date_part_second", ON, "SELECT date_part('SECOND', TIME'12:00:01.123456789') AS c", False),
    (
        "fields",
        ON,
        "SELECT hour(TIME'01:02:03.123456789'), minute(TIME'01:02:03.123456789'), "
        "second(TIME'01:02:03.123456789'), typeof(second(TIME'01:02:03.123456789'))",
        False,
    ),
    (
        "to_decimal",
        ON,
        "SELECT CAST(TIME'12:00:01.123456789' AS DECIMAL(20,9)) AS c, CAST(TIME'12:00:01.123456789' AS BIGINT) AS b",
        False,
    ),
    (
        "plus_interval",
        ON,
        "SELECT CAST(TIME'12:00:00.123456789' + INTERVAL '1' SECOND AS STRING) AS c, "
        "typeof(TIME'12:00:00.123456789' + INTERVAL '1' SECOND) AS t",
        False,
    ),
    ("minus_time", ON, "SELECT CAST(TIME'12:00:00.123456789' - TIME'12:00:00' AS STRING) AS c", False),
    ("time_diff", ON, "SELECT time_diff('MICROSECOND', TIME'12:00:00', TIME'12:00:00.000000999') AS c", False),
    ("time_diff_nanos", ON, "SELECT time_diff('NANOSECOND', TIME'12:00:00', TIME'12:00:00.000000123')", False),
    (
        "time_trunc",
        ON,
        "SELECT CAST(time_trunc('MICROSECOND', TIME'12:00:00.123456789') AS STRING) AS a, "
        "CAST(time_trunc('SECOND', TIME'12:00:00.123456789') AS STRING) AS b, "
        "typeof(time_trunc('SECOND', TIME'12:00:00.123456789')) AS t",
        False,
    ),
    # A cast to TIME(p) keeps p digits: from a string column and literal, and from a TIME.
    (
        "string_col_to_time1",
        ON,
        "SELECT CAST(CAST(s AS TIME(1)) AS STRING) AS a, time_to_micros(CAST(s AS TIME(1))) AS b "
        "FROM VALUES ('09:08:01.987654') t(s)",
        False,
    ),
    ("string_lit_to_time3", ON, "SELECT CAST(CAST('09:08:01.987654' AS TIME(3)) AS STRING) AS c", False),
    ("time6_to_time0", ON, "SELECT CAST(CAST(TIME'09:08:01.987654' AS TIME(0)) AS STRING) AS c", False),
    # EXTRACT(SECOND) of TIME(p), p <= 6.
    (
        "extract_second_low_precision",
        ON,
        "SELECT CAST(extract(SECOND FROM CAST('09:08:01.987654' AS TIME(0))) AS STRING) AS a, "
        "typeof(extract(SECOND FROM CAST('09:08:01.987654' AS TIME(0)))) AS ta, "
        "CAST(extract(SECOND FROM CAST('09:08:01.987654' AS TIME(3))) AS STRING) AS b, "
        "typeof(extract(SECOND FROM CAST('09:08:01.987654' AS TIME(3)))) AS tb, "
        "typeof(extract(SECOND FROM CAST('09:08:01.987654' AS TIME(6)))) AS tc",
        False,
    ),
    ("time_to_micros", ON, "SELECT time_to_micros(TIME'12:00:01.123456789') AS c", False),
    (
        "to_time",
        ON,
        "SELECT CAST(to_time('12:00:00.123456789') AS STRING) AS a, "
        "CAST(to_time('12:00:00.123456789', 'HH:mm:ss.SSSSSSSSS') AS STRING) AS b, "
        "typeof(to_time('12:00:00.123456789')) AS t",
        False,
    ),
    ("make_time", ON, "SELECT CAST(make_time(1, 2, 3.123456789) AS STRING) AS c", False),
    (
        "to_timestamp_ntz",
        ON,
        "SELECT CAST(CAST(TIME'12:00:01.123456789' AS TIMESTAMP_NTZ) AS STRING) LIKE '% 12:00:01.123456' AS c",
        False,
    ),
    (
        "make_timestamp",
        ON,
        "SELECT CAST(make_timestamp(DATE'2020-01-01', TIME'12:00:00.123456789') AS STRING) AS c",
        False,
    ),
    ("date_format", ON, "SELECT date_format(TIME'12:00:00.123456789', 'HH:mm:ss.SSSSSSSSS') AS c", False),
    # Nanosecond timestamp types: refused as Spark's default (spark.sql.timestampNanosTypes.enabled off).
    ("ts_ntz6", {}, "SELECT CAST(NULL AS TIMESTAMP_NTZ(6))", False),
    ("ts_ntz06", {}, "SELECT CAST(NULL AS TIMESTAMP_NTZ(06)) AS c", False),
    ("ts_ltz6", {}, "SELECT typeof(CAST(NULL AS TIMESTAMP_LTZ(6))) AS c", False),
    ("ts6", {}, "SELECT typeof(CAST(NULL AS TIMESTAMP(6))) AS c", False),
    ("ts6_ntz_default", NTZ, "SELECT typeof(CAST(NULL AS TIMESTAMP(6))) AS c", False),
    ("ts_ntz9", {}, "SELECT CAST(NULL AS TIMESTAMP_NTZ(9))", False),
    ("ts_ltz7", {}, "SELECT CAST(NULL AS TIMESTAMP_LTZ(7))", False),
    ("ts9", {}, "SELECT CAST(NULL AS TIMESTAMP(9))", False),
    ("ts_ntz0", {}, "SELECT CAST(NULL AS TIMESTAMP_NTZ(0))", False),
    ("ts_ntz3", {}, "SELECT CAST(NULL AS TIMESTAMP_NTZ(3))", False),
    ("ts_ltz5", {}, "SELECT CAST(NULL AS TIMESTAMP_LTZ(5))", False),
    ("ts0", {}, "SELECT CAST(NULL AS TIMESTAMP(0))", False),
    ("ts3_ntz_default", NTZ, "SELECT CAST(NULL AS TIMESTAMP(3))", False),
    ("ts_ntz10", {}, "SELECT CAST(NULL AS TIMESTAMP_NTZ(10))", False),
    ("ts_ntz_huge", {}, "SELECT CAST(NULL AS TIMESTAMP_NTZ(99999999999))", False),
    ("ts6_with_local", {}, "SELECT typeof(CAST(NULL AS TIMESTAMP(6) WITH LOCAL TIME ZONE)) AS c", False),
    ("ts3_with_local", {}, "SELECT CAST(NULL AS TIMESTAMP(3) WITH LOCAL TIME ZONE)", False),
    ("ts9_without", {}, "SELECT CAST(NULL AS TIMESTAMP(9) WITHOUT TIME ZONE)", False),
    ("ts6_without", {}, "SELECT typeof(CAST(NULL AS TIMESTAMP(6) WITHOUT TIME ZONE)) AS c", False),
    # The nanosecond types with spark.sql.timestampNanosTypes.enabled on (4.2: the key changes nothing).
    ("ts_ntz9_nanos_on", NANOS, "SELECT typeof(CAST(NULL AS TIMESTAMP_NTZ(9))) AS c", False),
    ("ts_ltz7_nanos_on", NANOS, "SELECT typeof(CAST(NULL AS TIMESTAMP_LTZ(7))) AS c", False),
    ("ts_ntz6_nanos_on", NANOS, "SELECT typeof(CAST(NULL AS TIMESTAMP_NTZ(6))) AS c", False),
    # A DDL schema string: a bad precision is the direct parse's PARSE_SYNTAX_ERROR.
    ("ddl_ts_ntz9", ON, "SELECT from_json('{\"a\":1}', 'a timestamp_ntz(9)')", False),
    ("ddl_ts9", ON, "SELECT from_json('{\"a\":1}', 'a timestamp(9)')", False),
    ("ddl_ts_ntz3", ON, "SELECT from_json('{\"a\":1}', 'a timestamp_ntz(3)')", False),
    ("ddl_time10", ON, "SELECT from_json('{\"a\":1}', 'a time(10)')", False),
    ("ddl_ts_ltz6", ON, "SELECT typeof(from_json('{\"a\":1}', 'a timestamp_ltz(6)')) AS c", False),
    ("ddl_ts6", ON, "SELECT typeof(from_json('{\"a\":1}', 'a timestamp(6)')) AS c", False),
    (
        "ddl_time3_value",
        ON,
        "SELECT CAST(from_json('{\"a\":\"12:00:00.123456789\"}', 'a time(3)').a AS STRING) AS c",
        False,
    ),
    ("ddl_time9", ON, "SELECT typeof(from_json('{\"a\":\"12:00:00.123456789\"}', 'a time(9)')) AS c", False),
    # Precision on the timestamp clocks.
    (
        "current_timestamp6",
        {},
        "SELECT current_timestamp(6), now(6), localtimestamp(6), "
        "typeof(current_timestamp(6)), typeof(localtimestamp(6))",
        True,
    ),
    ("current_timestamp_bigint", {}, "SELECT typeof(current_timestamp(CAST(6 AS BIGINT)))", True),
    ("current_timestamp9", {}, "SELECT current_timestamp(9)", False),
    ("current_timestamp_null", {}, "SELECT current_timestamp(CAST(NULL AS INT))", False),
    ("localtimestamp7", {}, "SELECT localtimestamp(7)", False),
    ("current_timestamp3", {}, "SELECT current_timestamp(3)", False),
    ("now10", {}, "SELECT now(10)", False),
    ("localtimestamp_negative", {}, "SELECT localtimestamp(-1)", False),
    ("current_timestamp_decimal", {}, "SELECT current_timestamp(6.0)", False),
    ("current_timestamp_string", {}, "SELECT current_timestamp('6')", False),
    # SQL-41: a literal given a collation by the implicit cast.
    ("instr_collated", {}, "SELECT instr(name COLLATE UTF8_LCASE, 'PL') FROM VALUES ('apple') t(name)", False),
    ("instr_literal_collated", {}, "SELECT instr(name, 'pl' COLLATE UTF8_LCASE) FROM VALUES ('apple') t(name)", False),
    ("locate_collated", {}, "SELECT locate('PL', name COLLATE UTF8_LCASE) FROM VALUES ('apple') t(name)", False),
    ("eq_collated", {}, "SELECT name COLLATE UTF8_LCASE = 'APPLE' FROM VALUES ('apple') t(name)", False),
]

CONDITION = re.compile(r"\[([A-Z0-9_.]+)\]")


def error_record(e: Exception) -> dict[str, Any]:
    """The condition and the message's first line, without the SQLSTATE and query context."""
    lines = [line for line in str(e).splitlines() if CONDITION.search(line)]
    line = lines[0] if lines else str(e).splitlines()[0]
    line = line[line.index("[") :] if "[" in line else line
    match = CONDITION.match(line)
    message = line.split(" SQLSTATE:")[0].strip()
    return {"error": match.group(1) if match else None, "message": message}


def run(spark: Any, conf: dict[str, str], sql: str, clock: bool) -> dict[str, Any]:
    from pyspark.sql import functions as F

    for k, v in conf.items():
        spark.conf.set(k, v)
    try:
        df = spark.sql(sql)
        names = list(df.columns)
        # Positional names first: two output names may be equal (Spark 4.2 cuts a literal's digits).
        positional = df.toDF(*[f"c{i}" for i in range(len(names))])
        rows = positional.select(*[F.col(f"c{i}").cast("string") for i in range(len(names))]).collect()
        if clock:
            typed = [[i, rows[0][i]] for i, n in enumerate(names) if n.startswith("typeof(")]
            return {"names": names, "typed": typed}
        return {"names": names, "values": [[r[i] for i in range(len(names))] for r in rows]}
    except Exception as e:  # a Spark error is the expected result of a case
        return error_record(e)
    finally:
        for k in conf:
            spark.conf.unset(k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote", help="a Spark Connect server (sc://host:port) instead of local mode")
    args = parser.parse_args()
    from pyspark.sql import SparkSession

    if args.remote:
        spark = SparkSession.builder.remote(args.remote).getOrCreate()
        spark.conf.set("spark.sql.session.timeZone", "UTC")
    else:
        spark = (
            SparkSession.builder.master("local[1]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
    cases = []
    for case_id, conf, sql, clock in CASES:
        cases.append({"id": case_id, "conf": conf, "sql": sql, "clock": clock, **run(spark, conf, sql, clock)})
    doc = {
        "format_version": FORMAT_VERSION,
        "generator": GENERATOR,
        "spark_version": spark.version,
        "session_time_zone": "UTC",
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    spark.stop()


if __name__ == "__main__":
    main()
