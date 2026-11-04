#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate what the LEGACY parser (`SimpleDateFormat`, `Locale.US`) needs from the JVM Spark 4.2.0 runs on.

* ``zone_strings``: `DateFormatSymbols.getZoneStrings()` in the JVM's order, the table a `z`/`Z` field
  searches (first row, then first name, matching a prefix of the text, ignoring case), each row with
  `TimeZone.getRawOffset()` and `getDSTSavings()` of its id;
* ``digit_zeros``: the zero of every run of decimal digits `Character.digit(c, 10)` reads in the Basic
  Multilingual Plane (`DecimalFormat` reads a field's digits one UTF-16 unit at a time);
* ``probes``: `try_to_timestamp` results under LEGACY with the session zone UTC, as epoch seconds (null
  when Spark gives NULL), for every zone name in January and July and for the edge cases the kernel tests.

Runs a REAL Spark (never from memory) in a throw-away container. Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_legacy_zone_strings.py \
        > tests/fixtures/spark-4.2.0/legacy-zone-strings.json

then `python3 tests/scripts/gen_legacy_zone_table.py`. The fixture goes to stdout; Spark logs go to stderr.
"""

import json
import sys

from pyspark.sql import SparkSession

ZONE_PATTERN = "yyyy-MM-dd HH:mm z"

# (text, pattern): the cases the kernel tests pin beyond the zone names.
EDGE_CASES = [
    ("2020-01-03-05", "YYYY-ww-MM-dd"),
    ("2020-01-03", "YYYY-ww-MM"),
    ("2020-01-02-5", "YYYY-ww-W"),
    ("2020-01-03-5", "YYYY-ww-W"),
    ("2020-01-02-1", "YYYY-ww-F"),
    ("2020-01-5", "YYYY-ww-u"),
    ("2020-01-005", "YYYY-ww-DDD"),
    ("2020-01-363", "YYYY-ww-DDD"),
    ("2020-53", "YYYY-ww"),
    ("2020-52", "YYYY-ww"),
    ("2019-2020-01", "yyyy-YYYY-ww"),
    ("2020-2019-01", "YYYY-yyyy-ww"),
    ("1500-02-29", "yyyy-MM-dd"),
    ("1500-02-29 Sat", "yyyy-MM-dd E"),
    ("1500-02-29 Mon", "yyyy-MM-dd E"),
    ("1500-02-29 7", "yyyy-MM-dd u"),
    ("1500-02-29 5", "yyyy-MM-dd F"),
    ("1500-02-29 1", "yyyy-MM-dd F"),
    ("1500-02-29 5", "yyyy-MM-dd W"),
    ("1500-02-29 9", "yyyy-MM-dd w"),
    ("1500-02-29 060", "yyyy-MM-dd DDD"),
    ("1500-02-29 061", "yyyy-MM-dd DDD"),
    ("1500 060", "yyyy DDD"),
    ("1500 060 Sat", "yyyy DDD E"),
    ("1500 060 Mon", "yyyy DDD E"),
    ("\u0662\u0660\u0662\u0664-\u0660\u0661-\u0660\u0662", "yyyy-MM-dd"),
    ("\uff12\uff10\uff12\uff14-\uff10\uff11-\uff10\uff12", "yyyy-MM-dd"),
    ("2024-01-02 \u0661\u0662:\u0663\u0664", "yyyy-MM-dd HH:mm"),
    ("\u0662\u0660\u0662\u0664\u0660\u0661\u0660\u0662", "yyyyMMdd"),
    ("2024-01-01 12:00 +\u0660\u0665\u0663\u0660", ZONE_PATTERN),
    ("2024-01-01 12:00 +0530", ZONE_PATTERN),
    ("2024-01-01 12:00 -0800", ZONE_PATTERN),
    ("2024-01-01 12:00 +0530", "yyyy-MM-dd HH:mm Z"),
    ("2024-01-01 12:00 +05:30", ZONE_PATTERN),
    ("2024-01-01 12:00 +05:30", "yyyy-MM-dd HH:mm Z"),
    ("2024-01-01 12:00 +05", ZONE_PATTERN),
    ("2024-01-01 12:00 +2400", ZONE_PATTERN),
    ("2024-01-01 12:00 +0560", ZONE_PATTERN),
    ("2024-01-01 12:00 GMT+05:30", ZONE_PATTERN),
    ("2024-01-01 12:00 GMT+5:30", ZONE_PATTERN),
    ("2024-01-01 12:00 gmt-08:00", ZONE_PATTERN),
    ("2024-01-01 12:00 GMT+0530", ZONE_PATTERN),
    ("2024-01-01 12:00 GMT", ZONE_PATTERN),
    ("2024-01-01 12:00 UTC+05:30", ZONE_PATTERN),
    ("2024-01-01 12:00 Z", ZONE_PATTERN),
    ("2024-01-01 12:00 Z", "yyyy-MM-dd HH:mm Z"),
    ("2024-01-01 12:00 America/Los_Angeles", ZONE_PATTERN),
    ("2024-07-01 12:00 pdt", ZONE_PATTERN),
    ("2024-07-01 12:00 pacific daylight time", ZONE_PATTERN),
    ("2024-07-01 12:00 India Standard Time", ZONE_PATTERN),
    ("2024-07-01 12:00 Israel Standard Time", ZONE_PATTERN),
    ("2024-07-01 12:00 " + "IST", ZONE_PATTERN),
    ("2024-07-01 12:00 PST8PDT", ZONE_PATTERN),
    ("2024-07-01 PST1200", "yyyy-MM-dd zHHmm"),
    ("2024-07-01 PSTfoo", "yyyy-MM-dd z'foo'"),
    ("2024-07-01 pst1200", "yyyy-MM-dd zHHmm"),
    ("2024-07-01 12:00 XYZ", ZONE_PATTERN),
]


def main() -> None:
    spark = SparkSession.builder.appName("legacy-zone-strings").getOrCreate()
    jvm = spark._jvm
    symbols = jvm.java.text.DateFormatSymbols.getInstance(jvm.java.util.Locale.US)
    rows = []
    for names in symbols.getZoneStrings():
        names = [None if n is None else str(n) for n in names]
        tz = jvm.java.util.TimeZone.getTimeZone(names[0])
        rows.append([*names[:5], int(tz.getRawOffset()), int(tz.getDSTSavings())])
    zeros = []
    c = 0
    while c < 0x10000:
        if jvm.java.lang.Character.digit(c, 10) == 0 and jvm.java.lang.Character.isDigit(c):
            zeros.append(c)
            c += 10
        else:
            c += 1

    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    cases = list(EDGE_CASES)
    names = sorted({n for r in rows for n in r[1:5] if n})
    for day in ("2024-01-15", "2024-07-01"):
        cases += [(f"{day} 12:00 {n}", ZONE_PATTERN) for n in names]
    df = spark.createDataFrame(cases, "t string, p string")
    got = df.selectExpr("t", "p", "cast(try_to_timestamp(t, p) as long) as s").collect()
    probes = [[r["t"], r["p"], r["s"]] for r in got]
    out = {
        "spark_version": spark.version,
        "java_version": str(jvm.java.lang.System.getProperty("java.version")),
        "zone_strings": rows,
        "digit_zeros": zeros,
        "probes": probes,
    }
    json.dump(out, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
