#!/usr/bin/env python3
"""Timestamp type and time zone labels against Spark 4.2 (Comet 1.2.0 correctness check 3).

Same wall clocks as TIMESTAMP (a session-zone instant) and TIMESTAMP_NTZ, crossed with the
conversions and functions where the label matters: the cast between the two, strings, dates,
truncation, zone conversion, arithmetic across a DST change, comparison and coalesce of the two
types, `typeof`, `date_format` with zone letters, and the constructors. The answer of a case is its
Spark type and rows, so a value with the right instant and the wrong type or zone fails. Run once
per zone; the fixture records the zone (`timeZone`). Usage (JDK as for `gen_expr_differential.py`):

    for z in Asia/Kolkata America/Los_Angeles; do
      env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \
          python -I tests/scripts/gen_timestamp_labels.py --zone $z \
          > tests/fixtures/spark-4.2.0/timestamp-labels-${z##*/}.json
    done
"""

import argparse
import importlib.util
import json
import pathlib
import sys

WALLS = [
    "1970-01-01 00:00:00",
    "2000-02-29 23:59:59.999999",
    "2024-03-10 01:30:00",  # the hour before the Los Angeles spring-forward gap
    "2024-03-10 02:30:00",  # inside that gap
    "2024-03-31 01:30:00",
    "2024-06-15 12:00:00",
    "2024-11-03 01:30:00",  # ambiguous in Los Angeles (fall back)
    "0001-01-01 00:00:00",
    None,
]


def setup_sql():
    rows = []
    for i, w in enumerate(WALLS):
        if w is None:
            rows.append(f"({i}, CAST(NULL AS TIMESTAMP), CAST(NULL AS TIMESTAMP_NTZ))")
        else:
            rows.append(f"({i}, CAST('{w}' AS TIMESTAMP), CAST('{w}' AS TIMESTAMP_NTZ))")
    return "CREATE OR REPLACE TEMPORARY VIEW t AS SELECT * FROM VALUES " + ", ".join(rows) + " AS t(id, ts, ntz)"


EXPRS = [
    "ts",
    "ntz",
    "CAST(ts AS TIMESTAMP_NTZ)",
    "CAST(ntz AS TIMESTAMP)",
    "CAST(ts AS STRING)",
    "CAST(ntz AS STRING)",
    "CAST(ts AS DATE)",
    "CAST(ntz AS DATE)",
    "CAST(ts AS BIGINT)",
    "CAST(ts AS DOUBLE)",
    "unix_timestamp(ts)",
    "unix_seconds(ts)",
    "unix_micros(ts)",
    "hour(ts)",
    "hour(ntz)",
    "minute(ts)",
    "date_trunc('hour', ts)",
    "date_trunc('day', ts)",
    "date_trunc('day', ntz)",
    "trunc(ts, 'MM')",
    "to_utc_timestamp(ts, 'Asia/Tokyo')",
    "from_utc_timestamp(ts, 'Asia/Tokyo')",
    "from_utc_timestamp(ntz, 'America/New_York')",
    "(ts + INTERVAL '1' DAY)",
    "(ts + INTERVAL '1' HOUR)",
    "(ntz + INTERVAL '1' DAY)",
    "(ntz + INTERVAL '1' HOUR)",
    "(ts - ts)",
    "(ntz - ntz)",
    "(ts = CAST(ntz AS TIMESTAMP))",
    "(ts < ntz)",
    "(ts = ntz)",
    "coalesce(ts, ntz)",
    "coalesce(ntz, ts)",
    "typeof(ts)",
    "typeof(ntz)",
    "typeof(coalesce(ts, ntz))",
    "typeof(ts + INTERVAL '1' DAY)",
    "typeof(ntz + INTERVAL '1' DAY)",
    "date_format(ts, 'yyyy-MM-dd HH:mm:ss')",
    "date_format(ts, 'yyyy-MM-dd HH:mm:ss z')",
    "date_format(ts, 'yyyy-MM-dd HH:mm:ss VV')",
    "date_format(ts, 'xxx')",
    "date_format(ntz, 'yyyy-MM-dd HH:mm:ss')",
    "timestamp_seconds(86400)",
    "timestamp_millis(86400000)",
    "timestamp_micros(86400000000)",
    "make_timestamp(2024, 3, 10, 2, 30, 0)",
    "make_timestamp(2024, 11, 3, 1, 30, 0)",
    "make_timestamp_ntz(2024, 3, 10, 2, 30, 0)",
    "to_timestamp('2024-03-10 01:30:00')",
    "to_timestamp_ntz('2024-03-10 01:30:00')",
    "TIMESTAMP'2024-06-15 12:00:00'",
    "TIMESTAMP_NTZ'2024-06-15 12:00:00'",
    "current_timezone()",
    "array(ts, ts)",
    "array(ntz, ntz)",
    "named_struct('a', ts, 'b', ntz)",
    "max(ts) OVER ()",
    "max(ntz) OVER ()",
    "min(ntz) OVER ()",
    "to_json(named_struct('a', ts, 'b', ntz))",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True)
    args = ap.parse_args()
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("gen_expr_differential", here / "gen_expr_differential.py")
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load gen_expr_differential.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("timestamp-labels")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.sql.session.timeZone", args.zone)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.datetime.java8API.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    hive = spark._jvm.org.apache.spark.sql.execution.HiveResult.hiveResultString
    setup = setup_sql()
    spark.sql(setup)
    cases = []
    for ansi in (True, False):
        spark.conf.set("spark.sql.ansi.enabled", "true" if ansi else "false")
        results = {}
        for lo in range(0, len(EXPRS), 16):
            results.update(base.run_batch(spark, hive, EXPRS[lo : lo + 16]))
        for e in EXPRS:
            case = {"sql": e, "ansi": ansi, "kids": [], "root": True}
            case.update(results[e])
            cases.append(case)
    json.dump({"spark": spark.version, "timeZone": args.zone, "setup": setup, "cases": cases}, sys.stdout, indent=0)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
