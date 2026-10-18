#!/usr/bin/env python3
"""Generate Spark differential fixtures for datetime format patterns (P2-T081a).

Runs a REAL Spark (never from memory) in a throw-away container. Formatting: `date_format` of fixed
instants under every pattern, in a UTC and a `+05:30` session. Parsing: `to_timestamp` and
`to_timestamp_ntz` of strings under the parse patterns, and the CSV reader's date and strict
`TIMESTAMP_NTZ` parsing (`from_csv` with `dateFormat` / `timestampNTZFormat`), with ANSI off (NULL on
a mismatch); the strings are Spark's own formatting of the instants plus hand-written edge cases and
seeded mutations. A pattern Spark rejects records its error class instead of values. Usage (from repo
root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[2]' /work/gen_datetime_pattern_goldens.py \
        > tests/fixtures/spark-4.2.0/datetime-patterns.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import contextlib
import json
import random
import re
import sys

from pyspark.sql import SparkSession

SEED = 20261019
ZONES = {"utc": "UTC", "ist": "+05:30"}
# Epoch microseconds of the instants formatted (UTC wall clock in the comment).
INSTANTS = [
    0,  # 1970-01-01 00:00:00
    1_577_934_245_123_456,  # 2020-01-02 03:04:05.123456
    1_593_615_600_000_000,  # 2020-07-01 15:00:00
    1_593_561_600_000_001,  # 2020-07-01 00:00:00.000001
    1_609_502_399_999_999,  # 2021-01-01 11:59:59.999999
    1_609_545_600_100_000,  # 2021-01-02 00:00:00.1
    -2_208_988_800_000_000,  # 1900-01-01
    -62_135_596_800_000_000,  # 0001-01-01
    -62_135_596_800_000_001,  # 0000-12-31 23:59:59.999999 (year 0 = 1 BC)
    253_402_300_799_999_999,  # 9999-12-31 23:59:59.999999
    253_402_300_800_000_000,  # +10000-01-01
    327_403_430_400_000_000,  # +12345-06-07
    -63_587_721_600_000_000,  # -0044-03-15
    951_782_400_000_000,  # 2000-02-29
    1_234_567_890_120_000,  # 2009-02-13 23:31:30.12
]
FORMAT_PATTERNS = [
    "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
    "yyyy-MM-dd'T'HH:mm:ss[.SSS]",
    "yyyy-MM-dd",
    "yyyy-MM-dd HH:mm:ss.SSSSSS",
    "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
    "y",
    "yy",
    "yyy",
    "yyyyy",
    "yyyyyy",
    "yyyyyyy",
    "u",
    "uu",
    "uuuu",
    "uuuuu",
    "G yyyy",
    "GGGG y",
    "GGGGG y",
    "G",
    "M",
    "MM",
    "MMM",
    "MMMM",
    "L",
    "LL",
    "LLL",
    "LLLL",
    "d",
    "dd",
    "D",
    "DD",
    "DDD",
    "Q",
    "QQ",
    "QQQ",
    "QQQQ",
    "q",
    "qq",
    "qqq",
    "qqqq",
    "E",
    "EE",
    "EEE",
    "EEEE",
    "a",
    "h",
    "hh",
    "K",
    "KK",
    "k",
    "kk",
    "H",
    "HH",
    "m",
    "mm",
    "s",
    "ss",
    "S",
    "SS",
    "SSS",
    "SSSS",
    "SSSSS",
    "SSSSSSSS",
    "SSSSSSSSS",
    "X",
    "XX",
    "XXX",
    "XXXX",
    "XXXXX",
    "x",
    "xx",
    "xxx",
    "xxxx",
    "xxxxx",
    "Z",
    "ZZ",
    "ZZZ",
    "ZZZZ",
    "ZZZZZ",
    "O",
    "OOOO",
    "VV",
    "z",
    "zz",
    "zzz",
    "zzzz",
    "'lit''eral'",
    "''",
    "'",
    "yyyy[-MM[-dd]]",
    "hh 'o''clock' a",
    "EEEEE",
    "MMMMM",
    "LLLLL",
    "ddd",
    "DDDD",
    "HHH",
    "hhh",
    "kkk",
    "KKK",
    "mmm",
    "sss",
    "aa",
    "SSSSSSSSSS",
    "Y",
    "YYYY",
    "w",
    "W",
    "e",
    "c",
    "A",
    "n",
    "N",
    "p",
    "F",
    "B",
    "#",
    "{",
    "}",
    "[",
    "]",
    "yyyy]",
    "yyyyMMdd",
    "yyyyMMddHHmmss",
    "dd/MM/yyyy",
    "MM/dd/yy",
    "yyyy/MM/dd HH:mm:ss.SSSSSS XXX",
    "uuuuu-MM",
    "dd.MM.yyyy HH:mm",
    "EEE, d MMM yyyy",
    "yyyy-DDD",
    "V",
    "VVV",
    "OO",
    "OOO",
    "XXXXXX",
    "xxxxxx",
    "ZZZZZZ",
    "zzzzz",
    "QQQQQ",
    "qqqqq",
    "t",
    "j",
    "r",
    "MMMM d, yyyy h:mm:ss a",
    "HH:mm:ss.SSS",
    "yyyy-MM-dd HH:mm:ss.SSS Z",
]
PARSE_PATTERNS = [
    "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
    "yyyy-MM-dd'T'HH:mm:ss[.SSS]",
    "yyyy-MM-dd",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd HH:mm:ss.SSSSSS",
    "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
    "yyyy-MM-dd HH:mm:ss.SSS",
    "y",
    "yy",
    "yyy",
    "yyyyy",
    "u",
    "uuuu",
    "G yyyy",
    "M",
    "MM",
    "MMM",
    "MMMM",
    "LLL",
    "d",
    "dd",
    "D",
    "DDD",
    "yyyy D",
    "yyyy-DDD",
    "yyyy-MM-dd D",
    "a",
    "h",
    "hh",
    "hh a",
    "hh:mm a",
    "K a",
    "k",
    "kk:mm",
    "H",
    "HH",
    "HH:mm",
    "m",
    "s",
    "S",
    "SSS",
    "SSSSSS",
    "SSSSSSSSS",
    "X",
    "XX",
    "XXX",
    "XXXX",
    "XXXXX",
    "x",
    "xxx",
    "Z",
    "ZZZZ",
    "ZZZZZ",
    "O",
    "OOOO",
    "VV",
    "z",
    "yyyy-MM-dd HH:mm X",
    "yyyy-MM-dd HH:mm Z",
    "yyyy-MM-dd HH:mm XXX",
    "yyyy-MM-dd HH:mm VV",
    "yyyy-MM-dd HH:mm z",
    "yyyy-MM-dd HH:mm O",
    "'lit''eral' yyyy",
    "yyyy[-MM[-dd]]",
    "yyyy-MM-dd[ HH:mm:ss]",
    "yyyyMMdd",
    "yyyyMMddHHmmss",
    "yyyyMMd",
    "dd/MM/yyyy",
    "MM/dd/yy",
    "MM/dd",
    "MM-dd HH:mm",
    "yyyy/MM/dd HH:mm:ss.SSSSSS XXX",
    "dd.MM.yyyy HH:mm",
    "d MMM yyyy",
    "MMMM d, yyyy h:mm:ss a",
    "HH:mm:ss.SSS",
    "E",
    "EEE, d MMM yyyy",
    "Q",
    "QQQ yyyy",
    "F",
    "e",
    "Y",
    "w",
    "A",
    "n",
]
EXTRA = {
    "yyyy-MM-dd": [
        "2020-01-01",
        "2020-1-1",
        " 2020-01-01",
        "2020-01-01 ",
        "12020-01-01",
        "+12020-01-01",
        "-2020-01-01",
        "+2020-01-01",
        "2020-02-30",
        "2020-02-29",
        "2021-02-29",
        "2020-13-01",
        "2020-00-10",
        "2020-01-32",
        "2020-01-01T10:00",
        "0000-01-01",
        "20-01-01",
        "02020-01-01",
        "x2020-01-01",
        "",
    ],
    "yyyy-MM-dd HH:mm:ss": [
        "2020-01-01 10:00:00",
        "2020-01-01",
        "2020-01-01 24:00:00",
        "2020-01-01 23:60:00",
        "2020-01-01 10:00:00.5",
        "2020-01-01 1:2:3",
        "2020-01-01  10:00:00",
        "2020-01-01T10:00:00",
    ],
    "yyyy-MM-dd HH:mm:ss.SSS": [
        "2020-01-01 10:00:00.1",
        "2020-01-01 10:00:00.12",
        "2020-01-01 10:00:00.123",
        "2020-01-01 10:00:00.1234",
        "2020-01-01 10:00:00",
        "2020-01-01 10:00:00.",
    ],
    "yyyy-MM-dd'T'HH:mm:ss[.SSS]": [
        "2020-01-01T10:00:00",
        "2020-01-01T10:00:00.1",
        "2020-01-01T10:00:00.123456",
        "2020-01-01T10:00:00.",
    ],
    "y": ["1", "2020", "-5", "+2020", "0", "123456789", "12345678901234567890"],
    "yy": ["20", "99", "00", "2020", "5"],
    "yyy": ["020", "2020", "5", "12345"],
    "yyyyy": ["02020", "2020", "12345", "+12345"],
    "M": ["1", "12", "13", "01", "001"],
    "MM": ["1", "01", "12"],
    "MMM": ["Jan", "jan", "JAN", "January", "Sep"],
    "MMMM": ["January", "january", "Jan", "September"],
    "LLL": ["Jan", "jan"],
    "d": ["1", "31", "32", "01"],
    "dd": ["1", "01", "31"],
    "D": ["1", "60", "366", "367", "001"],
    "DDD": ["001", "060", "1", "366"],
    "yyyy D": ["2020 366", "2021 366", "2020 60", "2021 60"],
    "yyyy-DDD": ["2020-060", "2020-60"],
    "yyyy-MM-dd D": ["2020-02-29 60", "2020-03-01 60"],
    "a": ["AM", "PM", "am", "pm", "A"],
    "h": ["0", "1", "12", "13"],
    "hh": ["01", "12", "1"],
    "hh a": ["12 AM", "12 PM", "01 PM", "11 am"],
    "hh:mm a": ["10:15 pm", "10:15 PM", "00:15 AM"],
    "K a": ["0 PM", "11 PM", "12 PM"],
    "k": ["24", "1", "0", "12"],
    "kk:mm": ["24:00", "01:30", "00:00"],
    "H": ["0", "23", "24", "7"],
    "HH": ["07", "7", "23"],
    "HH:mm": ["24:00", "23:59", "12:5"],
    "m": ["0", "59", "60"],
    "s": ["0", "59", "60"],
    "S": ["1", "12", "9"],
    "SSS": ["123", "12", "1", "1234"],
    "SSSSSS": ["123456", "1", "1234567"],
    "SSSSSSSSS": ["123456789", "1"],
    "X": ["Z", "+05", "+0530", "+05:30", "-08", "z"],
    "XX": ["Z", "+0530", "+05"],
    "XXX": ["Z", "+05:30", "+0530"],
    "XXXX": ["+0530", "+053015", "Z"],
    "XXXXX": ["+05:30:15", "+05:30", "Z"],
    "x": ["+05", "+00", "Z"],
    "xxx": ["+05:30", "+00:00", "Z"],
    "Z": ["+0530", "+0000", "Z", "-0800"],
    "ZZZZ": ["GMT+05:30", "GMT", "+05:30"],
    "ZZZZZ": ["+05:30", "Z"],
    "O": ["GMT+5", "GMT+5:30", "GMT"],
    "OOOO": ["GMT+05:30", "GMT"],
    "VV": ["UTC", "Z", "+05:30", "America/Los_Angeles", "Foo/Bar"],
    "z": ["UTC", "PST", "GMT", "+05:30"],
    "yyyy-MM-dd HH:mm X": ["2020-01-01 10:00 +05", "2020-01-01 10:00 Z", "2020-01-01 10:00"],
    "yyyy-MM-dd HH:mm Z": ["2020-01-01 10:00 +0530", "2020-01-01 10:00 -0800"],
    "yyyy-MM-dd HH:mm XXX": ["2020-01-01 10:00 +05:30", "2020-01-01 10:00 Z"],
    "yyyy-MM-dd HH:mm VV": ["2020-01-01 10:00 UTC", "2020-01-01 10:00 America/Los_Angeles", "2020-01-01 10:00 +01:00"],
    "yyyy-MM-dd HH:mm z": ["2020-01-01 10:00 UTC", "2020-01-01 10:00 PST", "2020-01-01 10:00 GMT"],
    "yyyy-MM-dd HH:mm O": ["2020-01-01 10:00 GMT+5", "2020-01-01 10:00 GMT"],
    "'lit''eral' yyyy": ["lit'eral 2020", "LIT'ERAL 2020", "literal 2020"],
    "yyyy[-MM[-dd]]": ["2020", "2020-05", "2020-05-06", "2020-05-"],
    "yyyy-MM-dd[ HH:mm:ss]": ["2020-05-06", "2020-05-06 10:11:12", "2020-05-06 10"],
    "yyyyMMdd": ["20200102", "2020012", "120200102", "202001020"],
    "yyyyMMddHHmmss": ["20200102030405"],
    "yyyyMMd": ["2020011", "20200111"],
    "MM/dd/yy": ["01/02/20", "1/2/20", "01/02/2020"],
    "MM/dd": ["02/29", "01/31"],
    "MM-dd HH:mm": ["02-28 10:00"],
    "d MMM yyyy": ["2 Jan 2020", "02 jan 2020"],
    "MMMM d, yyyy h:mm:ss a": ["January 2, 2020 3:04:05 AM", "march 9, 2021 12:00:00 pm"],
    "HH:mm:ss.SSS": ["10:11:12.345", "10:11:12"],
    "G yyyy": ["AD 2020", "BC 0044", "ad 2020"],
    "EEE, d MMM yyyy": ["Thu, 2 Jan 2020"],
    "QQQ yyyy": ["Q1 2020"],
    "E": ["Mon"],
    "Q": ["1"],
}
# One field per string: no separator, quote or comment character can occur in it.
CSV_OPTS = "'sep', '\\u0001', 'quote', '\\u0000', 'nullValue', '\\u0002'"
JUNK = [" ", "-", ":", ".", "T", "Z", "+", "0", "9", "x", "a"]


def collect(df, exprs):
    return [tuple(r) for r in df.selectExpr(*exprs).collect()]


def evaluate(spark, df, strings, expr):
    """`expr` over every string: a list of values, or the pattern's error when Spark rejects the pattern
    itself. A row that fails alone (a value out of range) records `{"error": class}` in its place."""
    try:
        return [r[1] for r in collect(df, ["s", expr])]
    except Exception as e:
        first = error_class(e)
    out = []
    for s in strings:
        one = spark.createDataFrame([(s,)], "s string")
        try:
            out.append(collect(one, ["s", expr])[0][1])
        except Exception as e:
            out.append({"error": error_class(e)})
    if all(isinstance(v, dict) for v in out):
        return {"error": first}
    return out


def error_class(e):
    m = re.search(r"\[([A-Z_]+(?:\.[A-Z_]+)?)\]", str(e))
    return m.group(1) if m else type(e).__name__


def main() -> None:
    spark = SparkSession.builder.appName("pattern-goldens").getOrCreate()
    spark.conf.set("spark.sql.ansi.enabled", "false")
    rng = random.Random(SEED)
    instants = spark.createDataFrame([(m,) for m in INSTANTS], "m long").selectExpr("m", "timestamp_micros(m) t")
    formats = []
    for p in FORMAT_PATTERNS:
        case = {"pattern": p}
        for key, zone in ZONES.items():
            spark.conf.set("spark.sql.session.timeZone", zone)
            try:
                rows = collect(instants, ["m", f"date_format(t, '{p.replace(chr(39), chr(92) + chr(39))}')"])
                case[key] = [v for _, v in rows]
            except Exception as e:
                case[key] = {"error": error_class(e)}
        formats.append(case)
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    parses = []
    for p in PARSE_PATTERNS:
        lit = p.replace("'", "\\'")
        strings = list(EXTRA.get(p, []))
        # A pattern Spark cannot format (`E`) adds no formatted strings.
        with contextlib.suppress(Exception):
            strings += [v for _, v in collect(instants, ["m", f"date_format(t, '{lit}')"]) if v is not None]
        for s in list(strings):
            for _ in range(2):
                i = rng.randint(0, len(s))
                strings.append(s[:i] + rng.choice([*JUNK, ""]) + s[i + rng.randint(0, 1) :])
        seen = []
        for s in strings:
            if s not in seen:
                seen.append(s)
        df = spark.createDataFrame([(s,) for s in seen], "s string")
        case = {"pattern": p, "strings": seen}
        exprs = {
            "ts_utc": f"unix_micros(to_timestamp(s, '{lit}'))",
            "ntz": f"unix_micros(cast(to_timestamp_ntz(s, '{lit}') as timestamp))",
            # `DateFormatter.parse` as the CSV and JSON readers use it (`to_date` is a cast of
            # `to_timestamp`, which moves values with an offset across midnight).
            "date": f"unix_date(from_csv(s, 'v date', map('dateFormat', '{lit}', {CSV_OPTS})).v)",
            # `parseWithoutTimeZone(_, allowTimeZone = false)`: a zone in the string is a mismatch.
            "ntz_strict": f"unix_micros(cast(from_csv(s, 'v timestamp_ntz', "
            f"map('timestampNTZFormat', '{lit}', {CSV_OPTS})).v as timestamp))",
        }
        for key, expr in exprs.items():
            case[key] = evaluate(spark, df, seen, expr)
        spark.conf.set("spark.sql.session.timeZone", "+05:30")
        case["ts_ist"] = evaluate(spark, df, seen, exprs["ts_utc"])
        spark.conf.set("spark.sql.session.timeZone", "UTC")
        parses.append(case)
    json.dump(
        {
            "spark_version": spark.version,
            "seed": SEED,
            "zones": ZONES,
            "instants": INSTANTS,
            "format": formats,
            "parse": parses,
        },
        sys.stdout,
        sort_keys=True,
        indent=0,
        ensure_ascii=False,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
