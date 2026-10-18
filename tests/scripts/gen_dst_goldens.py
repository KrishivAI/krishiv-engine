#!/usr/bin/env python3
"""Generate the DST-transition fixture for the datetime kernels (P2-T025a).

Runs a REAL Spark (never from memory) in a throw-away container under two session time zones with
daylight-saving transitions (America/Los_Angeles, Europe/Berlin). For UTC instants around each spring-forward
and fall-back transition (and a few ordinary ones) it records, in the session zone: the local fields (`hour`,
`minute`, `day`, `month`), the date, `date_add(date, 1)`, the instant moved by `INTERVAL '1' DAY`,
`'24' HOUR`, `'1' HOUR`, `'1' MONTH`, `'1' YEAR` and minus `'1' DAY`; for dates around each transition,
`CAST(date AS TIMESTAMP)` and `date + INTERVAL '36' HOUR`. All instants are epoch microseconds. Usage (from
repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_dst_goldens.py \
        > tests/fixtures/spark-4.2.0/dst-transitions.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import json
import sys

from pyspark.sql import SparkSession

SECOND = 1_000_000
# UTC seconds of each transition in 2024 (and the dates around them), per zone.
TRANSITIONS = {
    "America/Los_Angeles": [1710064800, 1730624400],  # 2024-03-10 10:00 UTC, 2024-11-03 09:00 UTC
    "Europe/Berlin": [1711846800, 1729990800],  # 2024-03-31 01:00 UTC, 2024-10-27 01:00 UTC
}
DATES = {
    "America/Los_Angeles": ["2024-03-09", "2024-03-10", "2024-03-11", "2024-11-02", "2024-11-03", "2024-11-04"],
    "Europe/Berlin": ["2024-03-30", "2024-03-31", "2024-04-01", "2024-10-26", "2024-10-27", "2024-10-28"],
}
OFFSETS = [-90_000, -43_200, -3_600, -1_800, -1, 0, 1, 1_800, 3_599, 3_600, 5_400, 43_200, 86_400]
EXPRESSIONS = {
    "hour": "extract(hour FROM t)",
    "minute": "extract(minute FROM t)",
    "day": "extract(day FROM t)",
    "month": "month(t)",
    "date": "unix_date(cast(t AS date))",
    "date_add": "unix_date(date_add(cast(t AS date), 1))",
    "plus_1_day": "unix_micros(t + INTERVAL '1' DAY)",
    "minus_1_day": "unix_micros(t - INTERVAL '1' DAY)",
    "plus_24_hours": "unix_micros(t + INTERVAL '24' HOUR)",
    "plus_1_hour": "unix_micros(t + INTERVAL '1' HOUR)",
    "plus_1_month": "unix_micros(t + INTERVAL '1' MONTH)",
    "plus_1_year": "unix_micros(t + INTERVAL '1' YEAR)",
}


def main() -> None:
    spark = SparkSession.builder.appName("dst-goldens").getOrCreate()
    cases = []
    for zone, transitions in TRANSITIONS.items():
        spark.conf.set("spark.sql.session.timeZone", zone)
        instants = sorted({(t + o) * SECOND + frac for t in transitions for o in OFFSETS for frac in (0, 999_999)})
        df = spark.createDataFrame([(x,) for x in instants], "x long").selectExpr("x", "timestamp_micros(x) t")
        cols = [f"{e} AS {k}" for k, e in EXPRESSIONS.items()]
        for row in df.selectExpr("x", *cols).collect():
            cases.append({"zone": zone, "kind": "instant", "x": row.x, **{k: getattr(row, k) for k in EXPRESSIONS}})
        ddf = spark.createDataFrame([(d,) for d in DATES[zone]], "s string").selectExpr("s", "cast(s as date) d")
        for row in ddf.selectExpr(
            "s",
            "unix_micros(cast(d AS timestamp)) AS to_timestamp",
            "unix_micros(d + INTERVAL '36' HOUR) AS plus_36_hours",
            "unix_micros(d + INTERVAL '24' HOUR) AS plus_24_hours",
            "unix_micros(d - INTERVAL '1' HOUR) AS minus_1_hour",
        ).collect():
            cases.append(
                {
                    "zone": zone,
                    "kind": "date",
                    "s": row.s,
                    "to_timestamp": row.to_timestamp,
                    "plus_36_hours": row.plus_36_hours,
                    "plus_24_hours": row.plus_24_hours,
                    "minus_1_hour": row.minus_1_hour,
                }
            )
    json.dump({"spark_version": spark.version, "cases": cases}, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
