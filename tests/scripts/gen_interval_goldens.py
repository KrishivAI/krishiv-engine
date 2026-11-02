#!/usr/bin/env python3
"""Generate the interval literal and arithmetic fixture (P2-T001b).

Every expectation comes from a REAL Spark 4.2 run in local mode: each statement's output name and type, its
single row cast to string, or the error condition it raises. The test is `crates/krishiv-session/tests/intervals.rs`.

Usage (from repo root):

    env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \
      python tests/scripts/gen_interval_goldens.py --out tests/fixtures/spark-4.2.0/interval-arithmetic.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.errors import PySparkException
from pyspark.sql import SparkSession

D = "date'2024-01-31'"
T = "timestamp'2024-01-31 00:00:00'"
N = "timestamp_ntz'2024-01-31 00:00:00'"

UTC = [
    # literals
    "interval '1 day'",
    "interval '1 day 2 hours'",
    "interval '1 month'",
    "interval '1 year 2 months'",
    "interval '1 hour'",
    "interval '1 week'",
    "interval '1 millisecond'",
    "interval '1 microsecond'",
    "interval '0 seconds'",
    "interval '0'",
    "interval '-1 day'",
    "- interval '1 day'",
    "interval -'1' day",
    "interval 1 day",
    "interval 1 day 2 hours",
    "interval 1 year 2 months",
    "interval 2 months",
    "interval 1 week",
    "interval 1 millisecond",
    "interval 1 second 1 millisecond",
    "interval 1.5 second",
    "interval 1.5 day",
    "interval '1' year",
    "interval '1 2' day to hour",
    "interval '1-2' year to month",
    "interval '1 2:3:4.5' day to second",
    "interval '1:2' hour to minute",
    "interval '10' hour",
    "interval '5.5' second",
    "interval 1 day 1 month",
    "interval '1 day 1 month'",
    "interval '1 month 0 day'",
    "interval 0 month 1 day",
    "interval '1 year -12 months'",
    "interval 'interval 1 day'",
    "interval '1day'",
    "interval '1 nanosecond'",
    "interval 1 nanosecond",
    "interval '1 hour 1 day'",
    "interval 1 second 1 day",
    "interval 1 week 1 hour",
    "interval '1 hour -60 minutes'",
    "interval 1 hour 3 millisecond",
    "interval '106751992' day",
    "interval 2147483647 month",
    "interval 1000000000000 hour",
    "interval '2147483647 days 25 hours'",
    "cast(interval '1' day as string)",
    "cast(interval '-1 2' day to hour as string)",
    "cast(interval '-1-2' year to month as string)",
    "cast(null as interval day)",
    "cast(null as interval)",
    "typeof(cast(null as interval hour to second))",
    "typeof(interval '1' second)",
    # date and timestamp with an interval
    f"{D} + interval '1' month",
    f"{D} - interval '1' month",
    f"interval '1' month + {D}",
    f"interval '1' hour + {D}",
    f"interval '1' day + {D}",
    f"{D} + interval '1' day",
    f"{D} - interval '1' day",
    f"{D} - interval '1' hour",
    f"{T} + interval '1' month",
    f"{T} - interval '1' month",
    f"{T} + interval '1' day",
    f"{T} - interval '1' day",
    f"interval '1' day + {T}",
    f"interval '1' month + {T}",
    f"{N} + interval '1' day",
    f"{N} - interval '1' month",
    f"{N} - interval '1 2' day to hour",
    "date'9999-12-31' + interval '1' day",
    # strings
    "'2024-01-31' + interval '1' day",
    "'2024-01-31' - interval '1' day",
    "interval '1' day + '2024-01-31'",
    "'2024-01-31' + interval '1' month",
    "'2024-01-31' - interval '1' month",
    "interval '1' month + '2024-01-31'",
    # NULLs
    f"{D} + null",
    f"null + {D}",
    f"{D} - null",
    f"null - {D}",
    "null + interval '1' month",
    "interval '1' month - null",
    "null - interval '1' day",
    "null * interval '1' day",
    "interval '1' day * null",
    "interval '1' day / null",
    "null / interval '1' day",
    # differences
    f"{D} - date'2024-01-01'",
    f"{T} - timestamp'2024-01-01 00:00:00'",
    f"{D} - timestamp'2024-01-01 00:00:00'",
    f"{T} - date'2024-01-01'",
    f"{N} - timestamp_ntz'2024-01-01 00:00:00'",
    f"{N} - timestamp'2024-01-01 00:00:00'",
    f"{T} - timestamp_ntz'2024-01-01 00:00:00'",
    f"{D} - timestamp_ntz'2024-01-01 00:00:00'",
    "'2024-01-31' - date'2024-01-01'",
    f"{D} - '2024-01-01'",
    "'2024-01-31' - timestamp'2024-01-01 00:00:00'",
    f"{T} - '2024-01-01'",
    "1 - date'2024-01-01'",
    "1 - timestamp'2024-01-01 00:00:00'",
    "timestamp'2024-01-01 00:00:00' - 1",
    # dates and timestamps with what is no interval
    "date'2020-01-30' + cast(5 as bigint)",
    "date'2020-01-30' + 5.5",
    "5.5 + date'2020-01-30'",
    "date'2020-01-30' - cast(5 as bigint)",
    "date'2020-01-30' + '1'",
    "'1' + date'2020-01-30'",
    "date'2020-01-30' - '1'",
    "date'2020-01-30' + date'2020-01-30'",
    "timestamp'2020-01-30 00:00:00' + 5",
    "5 + timestamp'2020-01-30 00:00:00'",
    "timestamp'2020-01-30 00:00:00' + '1'",
    "timestamp'2020-01-30 00:00:00' + timestamp'2020-01-30 00:00:00'",
    "timestamp'2020-01-30 00:00:00' + date'2020-01-30'",
    "date'2020-01-30' + timestamp'2020-01-30 00:00:00'",
    "timestamp'2020-01-30 00:00:00' + null",
    "null + timestamp'2020-01-30 00:00:00'",
    "date'2020-01-30' * 2",
    "timestamp'2020-01-30 00:00:00' * 2",
    # the resolver's fall-throughs
    "interval '1' day + 1",
    "1 + interval '1' day",
    "interval '1' month + 1",
    "1 + interval '1' month",
    "interval '1' day - 1",
    "1 - interval '1' day",
    "1 - interval '1' month",
    "interval '1' day + interval '1' month",
    "interval '1' month + interval '1' day",
    "interval '1' day - interval '1' month",
    "interval '1' month - interval '1' day",
    # within a family
    "interval '1' day + interval '2' hour",
    "interval '1' year + interval '2' month",
    "interval '1' day - interval '2' hour",
    "interval '178956970-7' year to month + interval '1' month",
    "interval '-178956970-8' year to month - interval '1' month",
    "- interval '1' month",
    "- interval '-2147483648' month",
    # multiply and divide
    "interval '1' day * 2",
    "2 * interval '1' day",
    "interval '1' day / 2",
    "2 / interval '1' day",
    "interval '1' day % 2",
    "interval '1' day * interval '1' day",
    "interval '1' day * '2'",
    "'2' * interval '1' day",
    "interval '1' day / '2'",
    "interval '1' month * 1.5",
    "interval '1' month * 1.5D",
    "interval '1' day * 2.5",
    "interval '1' day * 2Y",
    "interval '1' day * 2S",
    "interval '1' day * 2L",
    "interval '1' day * 2F",
    "interval '1' month / 2Y",
    "interval '1' month / 1.5",
    "interval '3' month / 2",
    "interval '1' day / 0",
    "interval '1' month / 0",
    "interval '1' month / 0D",
    "interval '1' month / 0.0",
    "interval '-2147483648' month / -1",
    "interval '-2147483648' month / -1D",
    "interval '-2147483648' month / -1.0",
    "interval '1' day * 9223372036854775807",
    "interval '2147483647' month * 2",
    "interval '2147483647' month * 2147483648L",
    "interval '1' month * cast('NaN' as double)",
    "interval '1' month / cast('inf' as double)",
    # try_*
    "try_add(interval '1' month, interval '2147483647' month)",
    f"try_add({D}, interval '1' month)",
    "try_add(interval '1' day, interval '1' hour)",
    "try_add(interval '1' year, interval '1' second)",
    "try_subtract(interval '-2147483648' month, interval '1' month)",
    "try_multiply(interval '2147483647' month, 2)",
    "try_divide(interval '1' month, 0)",
    "try_divide(interval '1' day, 2)",
    "+ interval '1' month",
    "positive(interval '1' day)",
    # comparison and common types
    "interval '1' day > interval '1' hour",
    "interval '1' day = interval '24' hour",
    "interval '1-1' year to month > interval '13' month",
    "interval '1' day = '1'",
    "interval '1' day = interval '1' month",
    "interval '1' day = 1",
    "case when true then interval '1' day else interval '2' hour end",
    "coalesce(interval '1' day, interval '2' hour)",
    "if(true, interval '1' year, interval '2' month)",
    "interval '1' day in (interval '24' hour)",
    # a cast to string renders the fields wherever the interval sits
    "cast(array(interval '1' day) as string)",
    "cast(named_struct('i', interval '1' day) as string)",
    "cast((select interval '1' day) as string)",
    "to_json(named_struct('i', interval '1' day))",
    "to_json(array(interval '1-2' year to month))",
    "to_csv(named_struct('i', interval '1' day))",
    # casts Spark refuses at analysis
    "cast(interval '1' day as double)",
    "cast(interval '1' day as boolean)",
    "cast(interval '1' day as timestamp)",
    "cast(interval '1' day as date)",
    "cast(interval '1' day as binary)",
    "cast(interval '1' day as interval month)",
    "cast(interval '1' month as interval day)",
    "cast(interval '1' month as double)",
    # within a family
    "cast(interval '1' day as interval hour)",
]

# Spark's answers differ with ANSI off: strings meet other types differently.
NON_ANSI = [
    "date'2020-01-30' + '1'",
    "'1' + date'2020-01-30'",
    "date'2020-01-30' - '1'",
    f"{D} - '2024-01-01'",
    "'2024-01-31' - date'2024-01-01'",
    "'2024-01-31' - timestamp'2024-01-01 00:00:00'",
    f"{T} - '2024-01-01'",
    "timestamp'2020-01-30 00:00:00' + '1'",
    "'2024-01-31' + interval '1' day",
    "'2024-01-31' - interval '1' day",
    "'2024-01-31' + interval '1' month",
    "'2024-01-31' - interval '1' month",
    "case when true then interval '1' day else '1' end",
    "interval '1' day * '2'",
    "'2' * interval '1' day",
    "interval '1' day / '2'",
    "interval '1' day = '1'",
    "interval '1' day in ('1')",
    "coalesce(interval '1' day, '1')",
    "interval '1' day + 1",
    "interval '1' day + interval '1' month",
    f"{D} + interval '1' month",
    f"{D} + interval '1' day",
    f"{D} + null",
    f"{D} - null",
    f"{D} - date'2024-01-01'",
    f"{T} - timestamp'2024-01-01 00:00:00'",
    "date'2020-01-30' + 5.5",
    "timestamp'2020-01-30 00:00:00' + 5",
    "date'9999-12-31' + interval '1' day",
    "interval '1' day / 0",
    "interval '1' month / 0",
    "interval '2147483647' month * 2",
    "interval '178956970-7' year to month + interval '1' month",
    "try_add(interval '1' month, interval '2147483647' month)",
    "try_divide(interval '1' month, 0)",
    "cast(interval '1' day as double)",
]

# Across a DST change the zone is what makes these differ.
LOS_ANGELES = [
    "timestamp'2024-03-11 00:00:00' - timestamp'2024-03-10 00:00:00'",
    "date'2024-03-11' - timestamp'2024-03-10 00:00:00'",
    "timestamp_ntz'2024-03-11 00:00:00' - timestamp'2024-03-10 00:00:00'",
    "timestamp'2024-03-11 00:00:00' - timestamp_ntz'2024-03-10 00:00:00'",
    "date'2024-03-11' - timestamp_ntz'2024-03-10 00:00:00'",
    "timestamp_ntz'2024-03-11 00:00:00' - timestamp_ntz'2024-03-10 00:00:00'",
    "timestamp'2024-03-09 01:30:00' + interval '1' day",
    "timestamp'2024-03-09 01:30:00' + interval '24' hour",
    "timestamp'2024-03-09 01:30:00' - interval '1' month",
    "timestamp_ntz'2024-03-09 02:30:00' + interval '1' day",
    "date'2024-03-10' + interval '3' hour",
]


def condition(e: PySparkException) -> str:
    """The error condition; Spark raises some arithmetic errors with none."""
    return e.getCondition() or "UNCLASSIFIED"


def run(spark: SparkSession, expr: str, zone: str, ansi: bool) -> dict:
    case: dict = {"expr": expr, "zone": zone, "ansi": ansi}
    try:
        df = spark.sql(f"SELECT {expr}")
    except PySparkException as e:
        case["error_class"] = condition(e)
        return case
    field = df.schema.fields[0]
    case["name"] = field.name
    case["type"] = field.dataType.simpleString()
    try:
        row = df.selectExpr(f"CAST(`{field.name.replace('`', '``')}` AS STRING)").collect()[0]
        case["value"] = row[0]
    except PySparkException as e:
        case["value_error_class"] = condition(e)
    except Exception as e:  # Spark raises a bare ArithmeticException for a multiply overflow
        case["value_error_class"] = "UNCLASSIFIED"
        case["value_error"] = str(e).split("\n")[0][:80]
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spark = (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    cases = []
    for zone, ansi, exprs in (
        ("UTC", True, UTC),
        ("America/Los_Angeles", True, LOS_ANGELES),
        ("UTC", False, NON_ANSI),
    ):
        spark.conf.set("spark.sql.session.timeZone", zone)
        spark.conf.set("spark.sql.ansi.enabled", str(ansi).lower())
        cases.extend(run(spark, e, zone, ansi) for e in exprs)
    doc = {
        "spark_version": spark.version,
        "generator": "tests/scripts/gen_interval_goldens.py",
        "cases": cases,
    }
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(cases)} cases against Spark {spark.version}")


if __name__ == "__main__":
    main()
