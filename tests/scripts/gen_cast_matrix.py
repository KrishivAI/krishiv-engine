#!/usr/bin/env python3
"""Generate the Spark 4.2 cast-matrix ground truth (P2-T021b).

For every source x target type pair of a representative type set, this asks a REAL Spark (never from memory):

* the analysis verdict of `CAST(CAST(NULL AS src) AS tgt)` under ANSI on, ANSI off and `TRY_CAST`
  (`analysis`: the resolved result type, or the error class and message parameters), and
* for each pair that analysis accepts, the result (or the error class, SQLSTATE and message parameters) of
  casting a set of edge values: boundaries, overflow, NaN and infinities, the ASCII and Unicode whitespace
  sets, Unicode digits, the empty string, malformed text and NULL (`cases`).

Value cases whose time-zone-dependence matters (anything touching TIMESTAMP, TIMESTAMP_NTZ) run under UTC and
America/Los_Angeles. `flag_cases` repeat chosen pairs under the legacy flags that change cast behaviour
(`spark.sql.legacy.castComplexTypesToString.enabled`, `spark.sql.legacy.charVarcharAsString`,
`spark.sql.legacy.allowNegativeScaleOfDecimal`, `spark.sql.timestampType`, `spark.sql.legacy.timeParserPolicy`).
Source values are built in SQL (`CAST('..' AS type)`, typed literals, constructors), so a source that Spark
itself cannot build is listed under `skipped_sources` and never guessed. A result value is the target value
rendered by `CAST(.. AS STRING)` (`hex()` for BINARY targets), or JSON null for a NULL result.

Each unit uses a fresh Spark Connect session on the shared oracle (temp-view-free, settings die with the
session). Usage (from repo root):

    SPARK_HOME=$HOME/krishiv-data/tmp/spark-home-stub uv run --no-project --python 3.12 \
        --with 'pyspark-client==4.2.0' python tests/scripts/gen_cast_matrix.py \
        > tests/fixtures/spark-4.2.0/cast-matrix.json

`--remote` overrides the endpoint (default sc://127.0.0.1:15012), `--only SRC:TGT` limits to one pair (debug),
`--workers N` sets the concurrency. Progress goes to stderr; the fixture to stdout.
"""

# ruff: noqa: RUF001, RUF005
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from pyspark.errors import PySparkException
from pyspark.sql import SparkSession

REMOTE = "sc://127.0.0.1:15012"
ZONES = ["UTC", "America/Los_Angeles"]
MODES = ["ansi_on", "ansi_off", "try_cast"]

# ---------------------------------------------------------------- types
DEC = ["decimal(1,0)", "decimal(5,2)", "decimal(10,0)", "decimal(18,6)", "decimal(38,10)", "decimal(38,38)"]
YM = ["interval year", "interval month", "interval year to month"]
DT = [
    "interval day",
    "interval hour",
    "interval minute",
    "interval second",
    "interval day to second",
    "interval hour to second",
    "interval minute to second",
]
NESTED_ARR = [
    "array<int>",
    "array<string>",
    "array<double>",
    "array<binary>",
    "array<array<int>>",
    "array<struct<a:int>>",
]
NESTED_MAP = ["map<string,int>", "map<int,array<int>>", "map<string,string>"]
NESTED_STRUCT = ["struct<a:int,b:string>", "struct<a:int>", "struct<a:int,b:string,c:int>", "struct<b:int,a:string>"]
SOURCES = (
    ["void", "boolean", "tinyint", "smallint", "int", "bigint", "float", "double"]
    + ["decimal(5,2)", "decimal(18,6)", "decimal(38,10)", "decimal(38,0)", "decimal(38,38)"]
    + ["string", "binary", "date", "timestamp", "timestamp_ntz", "variant"]
    + YM
    + DT
    + NESTED_ARR
    + NESTED_MAP
    + NESTED_STRUCT
)
TARGETS = (
    ["boolean", "tinyint", "smallint", "int", "bigint", "float", "double"]
    + DEC
    + ["string", "char(5)", "varchar(5)", "binary", "date", "timestamp", "timestamp_ntz", "variant"]
    + YM
    + DT
    + ["array<bigint>", "array<string>", "array<int>", "array<array<bigint>>", "array<struct<a:bigint>>"]
    + ["array<binary>", "array<boolean>", "map<string,bigint>", "map<int,string>", "map<string,string>"]
    + ["map<string,int>", "struct<a:bigint,b:string>", "struct<a:string,b:int>", "struct<a:int>"]
    + ["struct<x:int,y:string>", "struct<a:int,b:string,c:int>"]
)
NUMERIC = ["tinyint", "smallint", "int", "bigint", "float", "double", *DEC]
NUM_T = set(NUMERIC)
DT_T = {"date", "timestamp", "timestamp_ntz"}
TZ_TYPES = {"timestamp", "timestamp_ntz", "variant"}
GROUP = {}
for _t in NUMERIC:
    GROUP[_t] = "num"
for _t in YM:
    GROUP[_t] = "ym"
for _t in DT:
    GROUP[_t] = "dtv"
for _t in DT_T:
    GROUP[_t] = "dt"
GROUP["boolean"] = "bool"


def is_nested(t):
    return t.startswith(("array", "map", "struct"))


# ---------------------------------------------------------------- source values
def sql_lit(s):
    """Spark SQL string literal for `s`: printable ASCII verbatim, everything else as a \\u / \\U escape."""
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == "'":
            out.append("\\'")
        elif 0x20 <= o < 0x7F:
            out.append(ch)
        elif o <= 0xFFFF:
            out.append(f"\\u{o:04x}")
        else:
            out.append(f"\\U{o:08x}")
    return "'" + "".join(out) + "'"


def cast_vals(type_, texts):
    return [(t, f"CAST('{t}' AS {type_})") for t in texts]


def expr_vals(exprs):
    return [(e, e) for e in exprs]


WS = [
    "\x09", "\x0a", "\x0b", "\x0c", "\x0d", "\x1c", "\x1d", "\x1e", "\x1f", "\x20", "\x00", "\x01", "\x7f", "\x85",
    "\xa0", " ", " ", " ", " ", " ", "​", " ", " ", " ", " ",
    "　", "﻿", "᠎",
]  # fmt: skip
WS_BODIES = [
    ("1", "num"), ("1.5", "num"), ("true", "bool"), ("2020-01-01", "dt"), ("2020-01-01 01:02:03", "dt"),
    ("1-2", "ym"), ("1 02:03:04", "dtv"),
]  # fmt: skip
# Unicode-digit renderings of "123" and of dates / intervals.
UDIG = {
    "arabic_indic": "١٢٣",
    "ext_arabic_indic": "۱۲۳",
    "devanagari": "१२३",
    "fullwidth": "１２３",
    "thai": "๑๒๓",
    "mathematical_bold": "\U0001d7cf\U0001d7d0\U0001d7d1",
    "superscript": "¹²³",
    "circled": "①②③",
    "roman_numeral": "ⅠⅡⅢ",
    "vulgar_half": "½",
}
UDIG_DATE = [
    "٢٠٢٠-٠١-٠١",
    "２０２０-０１-０１",
    "2020-٠١-01",
    "２０２０-01-01 ０１:０２:０３",
]
UDIG_IV = ["١-٢", "１-２", "١ ٠٢:٠٣:٠٤"]

STR_GEN = [
    "", " ", "0", "1", "-1", "+1", "127", "128", "-128", "-129", "255", "32767", "32768", "-32768", "-32769",
    "2147483647", "2147483648", "-2147483648", "-2147483649", "9223372036854775807", "9223372036854775808",
    "-9223372036854775808", "-9223372036854775809", "1.5", "1.9", "-1.5", "-1.9", "2.5", ".5", "5.", "1e3", "1E3",
    "1e-3", "1.5e2", "0x10", "0b1", "1_000", "1,000", "12345678901234567890", "1.0e", "e1", "abc", "1a", "a1", "1 2",
    "1 1", "NaN", "nan", "NAN", "Infinity", "-Infinity", "+Infinity", "infinity", "INFINITY", "inf", "Inf", "-inf",
    "+inf", "1f", "1d", "1.5f", "1.5D", "1.5L", "3.4028235E38", "3.4028236E38", "3.5E38", "-3.4028235E38",
    "1.7976931348623157E308", "1.8E308", "1e309", "4.9E-324", "1e-400", "0.0", "-0.0", "00012", "0.1", "1e", "+", "-",
    ".", "+.", "..5", "1..5", "--1", "+-1", "123.456", "99999.999", "100000", "99999.994", "99999.995", "0.005",
    "0.015", "-0.005", "1.005", "123456789012345678901234567890123456789", "0.00000000000000000000000000000000000001",
    "1e38", "1e-38", "null", "NULL", "None", "n/a", "é", "今天", "\U0001f600", "'", '"', "\\", "a\x00b",
    "t", "T", "true", "TRUE", "True", "tRuE", "y", "Y", "yes", "YES", "1", "f", "F", "false", "FALSE", "False", "n",
    "N",
    "no", "NO", "0", "on", "off", "tr", "truee", "2", "-1", "10", "01", "00", "yes ", " no",
    "2020-01-01", "2020-1-1", "2020-01-01T00:00:00", "2020-01-01 junk", "2020-01-01x", "2020", "2020-01",
    "0001-01-01", "9999-12-31", "10000-01-01", "+10000-01-01", "-0001-01-01", "+999999-01-01", "2020-02-29",
    "2020-02-30", "2021-02-29", "2020-13-01", "2020-00-10", "2020-01-00", "2020-01-32", "20200101", "2020/01/01",
    "2020-01-01Z", "2020-01-01 ", "2020-01-01T", "2020-01-01 00", "T00:00:00", "1970-01-01", "epoch", "EPOCH",
    "2020-01-01 00:00:00", "2020-01-01T01:02:03.123456Z", "2020-01-01 01:02:03+08:00", "2020-01-01 01:02:03-0800",
    "2020-01-01 01:02:03.1234567", "2020-01-01 01:02:03.123456789", "2020-01-01 25:00:00", "2020-01-01 24:00:00",
    "2020-01-01 23:59:60", "2020-01-01 00:60:00", "2024-03-10 02:30:00", "2024-11-03 01:30:00",
    "2020-01-01 01:02:03 America/Los_Angeles", "2020-01-01 01:02:03 UTC", "2020-01-01 01:02:03 PST",
    "2020-01-01 01:02", "2020-01-01 1:2:3", "2020-01-01 01:02:03.", "2020-01-01 01:02:03.0",
    "1970-01-01 00:00:00.000001", "1969-12-31 23:59:59.999999", "0001-01-01 00:00:00", "9999-12-31 23:59:59.999999",
    "+294247-01-10 04:00:54.775807", "+294247-01-10 04:00:54.775808", "-290308-12-21 19:59:05.224192",
    "-290308-12-21 19:59:05.224191", "2020-01-01 01:02:03 +08:00", "2020-01-01 01:02:03Z", "2020-01-01 01:02:03z",
    "abcde", "abcdef", "abcdefghij", "ab ", " ab", "ab   ", "\u00e9" * 6, "1e39",
    "1-2", "+1-2", "-1-2", "178956970-7", "178956970-8", "-178956970-8", "-178956970-9", "1 2:3:4.5", "-1 2:3:4.5",
    "+1 2:3:4.5", "106751991 04:00:54.775807", "106751991 04:00:54.775808", "-106751991 04:00:54.775808", "1:2:3",
    "1:02:03.5", "INTERVAL '1-2' YEAR TO MONTH", "INTERVAL '1' YEAR", "interval 1 year", "INTERVAL '1' DAY",
    "interval '1 2:3:4' day to second", "1 year", "P1Y", "1-13", "0-0", "00:00:00", "2562047788:00:54.775807", "86400",
    "3600.5", "59.999999", "60", "5 6", "1 02", "-0-1", "0 0:0:0.0000001",
]  # fmt: skip


def string_cases():
    """(label, expr, kind): kind `gen` runs against every scalar target, other kinds only against the
    targets they are about (whitespace and Unicode-digit sets x their body type)."""
    out = [(sql_lit(s), sql_lit(s), "gen") for s in STR_GEN]
    for body, kind in WS_BODIES:
        for c in WS:
            for s in (c + body, body + c):
                out.append((sql_lit(s), sql_lit(s), "ws_" + kind))
        out.append((sql_lit(" " + body + " "), sql_lit(" " + body + " "), "ws_" + kind))
        out.append((sql_lit(" " + body + " "), sql_lit(" " + body + " "), "ws_" + kind))
        out.append((sql_lit("\n\t " + body + " \t\n"), sql_lit("\n\t " + body + " \t\n"), "ws_" + kind))
    for d in UDIG.values():
        for s in (d, "+" + d, "-" + d, d + ".5", "1" + d, d + "e1", d + "d", "0x" + d):
            out.append((sql_lit(s), sql_lit(s), "ud_num"))
    for s in UDIG_DATE:
        out.append((sql_lit(s), sql_lit(s), "ud_dt"))
    for s in UDIG_IV:
        out.append((sql_lit(s), sql_lit(s), "ud_iv"))
    out.append(("١ as true", sql_lit("١"), "ud_bool"))
    out.append(("ｔrue", sql_lit("ｔrue"), "ud_bool"))
    seen, uniq = set(), []
    for label, expr, kind in out:
        if (label, kind) not in seen:
            seen.add((label, kind))
            uniq.append((label, expr, kind))
    return uniq


def string_kind_applies(kind, tgt):
    if kind == "gen":
        return True
    g = GROUP.get(tgt)
    if kind.startswith("ws_"):
        k = kind[3:]
        if k == "num":
            return g in ("num", "bool")
        return g == k
    if kind == "ud_num":
        return g == "num"
    if kind == "ud_dt":
        return g == "dt"
    if kind == "ud_iv":
        return g in ("ym", "dtv")
    if kind == "ud_bool":
        return g == "bool"
    return False


def max_dec(p, s):
    return "0." + "9" * 38 if (p, s) == (38, 38) else "9" * (p - s) + ("." + "9" * s if s else "")


def source_values():
    v = {}
    v["void"] = [("NULL", "NULL")]
    v["boolean"] = expr_vals(["true", "false"])
    v["tinyint"] = cast_vals("tinyint", ["-128", "127", "0", "1", "-1", "100"])
    v["smallint"] = cast_vals("smallint", ["-32768", "32767", "0", "1", "-1", "128", "-129", "255", "256"])
    v["int"] = cast_vals(
        "int",
        ["-2147483648", "2147483647", "0", "1", "-1", "127", "128", "-129", "32767", "32768", "-32769", "65536"]
        + ["16777217"],
    )
    v["bigint"] = cast_vals(
        "bigint",
        ["-9223372036854775808", "9223372036854775807", "0", "1", "-1", "2147483647", "2147483648", "-2147483649"]
        + ["4294967296", "9007199254740993", "16777217"],
    )
    v["float"] = cast_vals(
        "float",
        ["NaN", "Infinity", "-Infinity", "0.0", "-0.0", "1.5", "-1.5", "2.5", "0.5", "1.0E10", "2.14748365E9"]
        + ["9.223372E18", "3.4028235E38", "-3.4028235E38", "1.4E-45", "1.17549435E-38", "16777216.0", "123.456"]
        + ["1.0E20", "0.1", "127.9", "128.0", "32768.0"],
    )
    v["double"] = cast_vals(
        "double",
        ["NaN", "Infinity", "-Infinity", "0.0", "-0.0", "1.5", "-1.5", "2.5", "0.5", "-0.5", "1.0E10"]
        + ["2147483647.0", "2147483647.5", "2147483648.0", "-2147483648.0", "-2147483648.9", "-2147483649.0"]
        + ["9.223372036854775807E18", "9.3E18", "-9.3E18", "1.0E20", "1.7976931348623157E308", "4.9E-324"]
        + ["1.0E-10", "0.1", "123.456", "3.4028235E38", "3.5E38", "127.9", "128.0", "-128.9", "32767.5", "32768.0"]
        + ["99999.995", "1.005", "100000.0", "1.0E39", "-129.0", "255.0"],
    )
    v["decimal(5,2)"] = cast_vals(
        "decimal(5,2)",
        ["0.00", "1.00", "-1.00", "123.45", "999.99", "-999.99", "0.01", "0.50", "1.50", "2.50", "-0.50", "-2.50"]
        + ["127.00", "128.00", "-129.00", "255.00"],
    )
    v["decimal(18,6)"] = cast_vals(
        "decimal(18,6)",
        ["0", "1.5", "123456789012.345678", "-123456789012.345678", "2147483647.000000", "2147483648.000000"]
        + ["922337203685.477580", "0.000001", "999999999999.999999"],
    )
    v["decimal(38,10)"] = cast_vals(
        "decimal(38,10)",
        ["0", "1.5", "-1.5", "12345678901234567890123456.7890123456", max_dec(38, 10), "-" + max_dec(38, 10)]
        + ["9223372036854775807", "9223372036854775808", "9223372036854775807.9999999999", "2147483647.5"]
        + ["0.0000000001", "0.0000000005", "1234567890123456789012345678"],
    )
    v["decimal(38,0)"] = cast_vals(
        "decimal(38,0)",
        [max_dec(38, 0), "-" + max_dec(38, 0), "9223372036854775807", "9223372036854775808", "-9223372036854775808"]
        + ["-9223372036854775809", "4294967296", "0", "1", "2147483648", "128", "65536"],
    )
    v["decimal(38,38)"] = cast_vals(
        "decimal(38,38)", [max_dec(38, 38), "-0.5", "0.5", "0." + "0" * 37 + "1", "0.5" + "0" * 35]
    )
    v["string"] = []  # filled from string_cases()
    v["binary"] = expr_vals(
        ["X''", "X'00'", "X'41'", "X'414243'", "X'FF'", "X'80'", "X'C3A9'", "X'E282AC'", "X'F09F9880'", "X'C328'"]
        + ["X'EDA080'", "X'20'", "X'3120'", "X'31'", "X'74727565'", "X'7B7D'", "X'0000000000000001'", "X'FF'"]
    )
    v["date"] = expr_vals(
        ["DATE'1970-01-01'", "DATE'0001-01-01'", "DATE'9999-12-31'", "DATE'2024-02-29'", "DATE'1969-12-31'"]
        + ["DATE'2020-01-01'", "CAST('+10000-01-01' AS DATE)", "CAST('-0001-01-01' AS DATE)", "DATE'2024-03-10'"]
        + ["CAST('+999999-12-31' AS DATE)"]
    )
    ts = [
        "1970-01-01 00:00:00", "2024-03-10 02:30:00", "2024-03-10 03:30:00", "2024-11-03 01:30:00",
        "0001-01-01 00:00:00", "9999-12-31 23:59:59.999999", "2020-06-15 12:34:56.789012",
        "1969-12-31 23:59:59.999999", "2020-01-01 00:00:00",
    ]  # fmt: skip
    v["timestamp"] = expr_vals([f"TIMESTAMP'{t}'" for t in ts]) + expr_vals(
        ["timestamp_micros(9223372036854775807)", "timestamp_micros(-9223372036854775808)"]
        + ["timestamp_micros(2147483647000000)", "timestamp_micros(2147483648000000)", "timestamp_micros(-1)"]
        + ["timestamp_micros(1)", "timestamp_micros(1500000)", "timestamp_micros(-1500000)"]
        + ["timestamp_seconds(127)", "timestamp_seconds(128)", "timestamp_seconds(32768)"]
    )
    v["timestamp_ntz"] = expr_vals([f"TIMESTAMP_NTZ'{t}'" for t in ts]) + expr_vals(
        [
            "CAST(timestamp_micros(9223372036854775807) AS TIMESTAMP_NTZ)",
            "CAST(timestamp_micros(1500000) AS TIMESTAMP_NTZ)",
        ]
        + ["CAST(timestamp_micros(-1500000) AS TIMESTAMP_NTZ)"]
    )
    v["variant"] = expr_vals(
        ["parse_json('null')", "parse_json('true')", "parse_json('false')", "parse_json('1')", "parse_json('-1')"]
        + ["parse_json('128')", "parse_json('9223372036854775807')", "parse_json('1.5')", "parse_json('\"abc\"')"]
        + ["parse_json('\"2020-01-01\"')", "parse_json('\"1\"')", "parse_json('[1,2]')", "parse_json('{\"a\":1}')"]
        + ["parse_json('3.14159265358979323846')", "parse_json('1e300')", "parse_json('\"2020-01-01 01:02:03\"')"]
        + ["parse_json('\"true\"')", "parse_json('\"\"')", "parse_json('99999999999999999999')"]
    )
    v["interval year"] = expr_vals(
        [f"INTERVAL '{x}' YEAR" for x in ("0", "1", "-1", "178956970", "-178956970", "178956971")]
    )
    v["interval month"] = expr_vals(
        [f"INTERVAL '{x}' MONTH" for x in ("0", "13", "-13", "2147483647", "-2147483648", "127", "128")]
    )
    v["interval year to month"] = expr_vals(
        [f"INTERVAL '{x}' YEAR TO MONTH" for x in ("0-0", "1-2", "-1-2", "178956970-7", "-178956970-8", "10-11")]
    )
    v["interval day"] = expr_vals([f"INTERVAL '{x}' DAY" for x in ("0", "1", "-1", "106751991", "-106751991")])
    v["interval hour"] = expr_vals(
        [f"INTERVAL '{x}' HOUR" for x in ("0", "1", "25", "-25", "2562047788", "128", "32768")]
    )
    v["interval minute"] = expr_vals(
        [f"INTERVAL '{x}' MINUTE" for x in ("0", "1", "61", "-61", "153722867280", "128", "32768")]
    )
    v["interval second"] = expr_vals(
        [f"INTERVAL '{x}' SECOND" for x in ("0", "1", "1.5", "-1.5", "59.999999", "61", "3600", "9223372036854.775807")]
        + [f"INTERVAL '{x}' SECOND" for x in ("128", "2147483648", "0.000001", "-0.5")]
    )
    v["interval day to second"] = expr_vals(
        [f"INTERVAL '{x}' DAY TO SECOND" for x in ("0 00:00:00", "1 02:03:04.567891", "-1 02:03:04.567891")]
        + [f"INTERVAL '{x}' DAY TO SECOND" for x in ("106751991 04:00:54.775807", "-106751991 04:00:54.775808")]
        + [f"INTERVAL '{x}' DAY TO SECOND" for x in ("0 00:00:00.5", "0 00:00:01.999999", "128 00:00:00")]
    )
    v["interval hour to second"] = expr_vals(
        [f"INTERVAL '{x}' HOUR TO SECOND" for x in ("00:00:00", "25:03:04.5", "-25:03:04.5", "2562047788:00:54.775807")]
    )
    v["interval minute to second"] = expr_vals(
        [f"INTERVAL '{x}' MINUTE TO SECOND" for x in ("00:00", "61:04.5", "-61:04.5", "153722867280:54.775807")]
    )
    v["array<int>"] = expr_vals(
        ["array(1, 2, 3)", "array(1, NULL, 3)", "CAST(array() AS array<int>)", "array(2147483647, -2147483648)"]
        + ["array(128, 32768)", "array(0, 1, -1)"]
    )
    v["array<string>"] = expr_vals(
        ["array('1', 'x', NULL)", "array('2147483648')", "array(' 1 ')", "array('true', 'false')", "array('')"]
        + ["array('2020-01-01')", "array('\\u00a01')"]
    )
    v["array<double>"] = expr_vals(
        ["array(1.5D, 2.5D)", "array(CAST('NaN' AS DOUBLE), CAST('Infinity' AS DOUBLE))", "array(1.0E10D, -1.0E10D)"]
        + ["array(CAST(NULL AS DOUBLE))"]
    )
    v["array<binary>"] = expr_vals(["array(X'41', X'FF')", "array(X'', NULL)", "array(X'C3A9')"])
    v["array<array<int>>"] = expr_vals(
        ["array(array(1, 2), array(3))", "array(array(1, NULL), NULL)", "array(array(2147483647))"]
    )
    v["array<struct<a:int>>"] = expr_vals(
        ["array(named_struct('a', 1), NULL)", "array(named_struct('a', 2147483647))", "array(named_struct('a', NULL))"]
    )
    v["map<string,int>"] = expr_vals(
        ["map('a', 1, 'b', NULL)", "CAST(map() AS map<string,int>)", "map('1', 2147483647)", "map('k', 128)"]
    )
    v["map<int,array<int>>"] = expr_vals(
        ["map(1, array(1, 2))", "map(128, array(NULL))", "map(1, CAST(NULL AS array<int>))"]
    )
    v["map<string,string>"] = expr_vals(["map('k', '1')", "map(' k', 'x')", "map('1', '2147483648')", "map('a', NULL)"])
    v["struct<a:int,b:string>"] = expr_vals(
        [
            "named_struct('a', 1, 'b', 'x')",
            "named_struct('a', NULL, 'b', NULL)",
            "named_struct('a', 2147483647, 'b', '2147483648')",
        ]
        + ["named_struct('a', 128, 'b', ' 1 ')"]
    )
    v["struct<a:int>"] = expr_vals(["named_struct('a', 1)", "named_struct('a', NULL)"])
    v["struct<a:int,b:string,c:int>"] = expr_vals(["named_struct('a', 1, 'b', 'x', 'c', 3)"])
    v["struct<b:int,a:string>"] = expr_vals(["named_struct('b', 1, 'a', 'x')"])
    for t in list(v):
        if t != "void":
            typ = t
            v[t].append(("NULL", f"CAST(NULL AS {typ})"))
    return v


# ---------------------------------------------------------------- oracle plumbing
def error_dict(e):
    out = {"exception": type(e).__name__}
    if isinstance(e, PySparkException):
        cls = e.getErrorClass()
        if cls:
            out["class"] = cls
            params = e.getMessageParameters() or {}
            out["params"] = {k: params[k] for k in sorted(params)}
        state = e.getSqlState()
        if state:
            out["sqlstate"] = state
    if "class" not in out:
        out["message"] = str(e).strip().split("\n")[0][:300]
    return out


def sql_for(mode, expr, tgt):
    if mode == "try_cast":
        return f"TRY_CAST({expr} AS {tgt})"
    return f"CAST({expr} AS {tgt})"


def needs_hex(src, tgt):
    """Results that may hold bytes that are not UTF-8 (any binary involved) are returned as hex of their text."""
    return tgt != "binary" and "binary" in f"{src} {tgt}"


def render(mode, expr, tgt, hexed=False):
    inner = sql_for(mode, expr, tgt)
    if tgt == "binary":
        return f"hex({inner})"
    if hexed:
        return f"hex(CAST(CAST({inner} AS STRING) AS BINARY))"
    return f"CAST({inner} AS STRING)"


def session(remote, conf):
    s = SparkSession.builder.remote(remote).create()
    for k, v in conf.items():
        s.conf.set(k, v)
    return s


def run_batch(s, mode, src, tgt, items):
    """items [(label, expr)] -> {label: {"v": str|None} | error dict}; bisects batches that raise."""
    out = {}
    hexed = needs_hex(src, tgt)
    key = "hex" if hexed else "v"

    def go(chunk):
        cols = ", ".join(f"{render(mode, e, tgt, hexed)} AS c{i}" for i, (_, e) in enumerate(chunk))
        try:
            row = s.sql(f"SELECT {cols}").collect()[0]
        except Exception as e:
            if len(chunk) == 1:
                out[chunk[0][0]] = {"error": error_dict(e)}
                return
            mid = len(chunk) // 2
            go(chunk[:mid])
            go(chunk[mid:])
            return
        for i, (label, _) in enumerate(chunk):
            out[label] = {key: row[i]}

    for i in range(0, len(items), 120):
        go(items[i : i + 120])
    return out


def analysis_one(remote, conf, mode, src, tgt):
    s = session(remote, conf)
    try:
        src_expr = "NULL" if src == "void" else f"CAST(NULL AS {src})"
        try:
            row = s.sql(f"SELECT typeof({sql_for(mode, src_expr, tgt)}) AS t").collect()[0]
            return {"ok": row[0]}
        except Exception as e:
            return {"error": error_dict(e)}
    finally:
        s.stop()


def mode_conf(mode, tz="UTC", extra=None):
    conf = {"spark.sql.ansi.enabled": "false" if mode == "ansi_off" else "true", "spark.sql.session.timeZone": tz}
    conf.update(extra or {})
    return conf


def run_unit(remote, mode, tz, src, tgt, items, extra=None):
    s = session(remote, mode_conf(mode, tz, extra))
    try:
        return run_batch(s, mode, src, tgt, items)
    finally:
        s.stop()


def check_sources(remote, vals):
    """Evaluate every source expression alone; return usable values and the ones Spark cannot build."""
    ok, skipped = {}, []
    s = session(remote, mode_conf("ansi_on"))
    try:
        for src, items in vals.items():
            ok[src] = []
            for label, expr in items:
                try:
                    s.sql(f"SELECT hex(CAST(CAST({expr} AS STRING) AS BINARY)) AS x").collect()
                    ok[src].append((label, expr))
                except Exception as e:
                    skipped.append({"src": src, "value": label, "error": error_dict(e)})
    finally:
        s.stop()
    return ok, skipped


# ---------------------------------------------------------------- flag cases
def flag_plan(vals):
    """[(flag name, conf, [(src, tgt, [(label, expr)])])]"""
    nested_src = [t for t in SOURCES if is_nested(t)]
    cplx = [(s, "string", vals[s]) for s in nested_src] + [
        ("array<int>", "varchar(5)", vals["array<int>"]),
        ("struct<a:int,b:string>", "char(30)", vals["struct<a:int,b:string>"]),
    ]
    chars = [("string", f"{k}(5)", _char_strings()) for k in ("char", "varchar")]
    chars += [(s, "varchar(3)", vals[s]) for s in ("int", "bigint", "double", "boolean", "date", "decimal(5,2)")]
    negs = [(e, e) for e in ("1E3BD", "12E3BD", "1.5E4BD", "99E5BD", "1E-3BD", "123E-1BD", "0E3BD", "-7E2BD")]
    neg_tgts = ["tinyint", "int", "bigint", "double", "string", "decimal(10,2)", "decimal(3,0)", "boolean", "timestamp"]
    neg = [("decimal(neg-scale literal)", t, negs) for t in neg_tgts]
    neg += [("int", "decimal(5,-2)", vals["int"][:3]), ("string", "decimal(3,-1)", vals["string"][:3])]
    ntz = [
        (s, "timestamp", vals[s])
        for s in ("string", "date", "timestamp_ntz", "timestamp", "int", "bigint", "double", "decimal(38,10)")
    ] + [("timestamp", "string", vals["timestamp"]), ("timestamp_ntz", "timestamp", vals["timestamp_ntz"])]
    parser = [(s, t, _parser_strings()) for s in ("string",) for t in ("date", "timestamp", "timestamp_ntz")]
    return [
        ("castComplexTypesToString", {"spark.sql.legacy.castComplexTypesToString.enabled": "true"}, cplx),
        ("charVarcharAsString", {"spark.sql.legacy.charVarcharAsString": "true"}, chars),
        ("charVarcharAsString_default", {"spark.sql.legacy.charVarcharAsString": "false"}, chars),
        ("allowNegativeScaleOfDecimal", {"spark.sql.legacy.allowNegativeScaleOfDecimal": "true"}, neg),
        ("allowNegativeScaleOfDecimal_default", {"spark.sql.legacy.allowNegativeScaleOfDecimal": "false"}, neg),
        ("timestampType_NTZ", {"spark.sql.timestampType": "TIMESTAMP_NTZ"}, ntz),
        ("timeParserPolicy_LEGACY", {"spark.sql.legacy.timeParserPolicy": "LEGACY"}, parser),
        ("timeParserPolicy_EXCEPTION", {"spark.sql.legacy.timeParserPolicy": "EXCEPTION"}, parser),
    ]


def _char_strings():
    return [
        (sql_lit(s), sql_lit(s)) for s in ["", "a", "abcde", "abcdef", "abcdefghij", "ab ", " ab", "ab   ", "éééééé"]
    ] + [("NULL", "CAST(NULL AS STRING)")]


def _parser_strings():
    xs = [
        "2020-01-01",
        "2020-1-1",
        "2020-01-01 01:02:03",
        "2020-01-01T01:02:03Z",
        " 2020-01-01 ",
        "2020-02-30",
        "20200101",
        "junk",
    ]
    return [(sql_lit(s), sql_lit(s)) for s in xs]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--only", help="SRC:TGT, limit to one pair (debug)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    probe = SparkSession.builder.remote(args.remote).create()
    version = probe.version
    probe.stop()

    vals = source_values()
    scases = string_cases()
    vals["string"] = [(label, expr) for label, expr, _ in scases] + [("NULL", "CAST(NULL AS STRING)")]
    kinds = {label: kind for label, _, kind in scases}
    vals, skipped = check_sources(args.remote, vals)
    print(f"sources checked, skipped {len(skipped)}", file=sys.stderr)

    pairs = [(s, t) for s in SOURCES for t in TARGETS]
    if args.only:
        a, b = args.only.split(":")
        pairs = [(a, b)]
    pool = ThreadPoolExecutor(max_workers=args.workers)

    # analysis verdicts
    futs = {(m, s, t): pool.submit(analysis_one, args.remote, mode_conf(m), m, s, t) for m in MODES for s, t in pairs}
    analysis = {m: {} for m in MODES}
    for (m, s, t), f in futs.items():
        analysis[m][f"{s} -> {t}"] = f.result()
    print("analysis done", file=sys.stderr)

    # value cases
    cases = {}  # (src, tgt, value, tz) -> {mode: result}
    work = []
    for s, t in pairs:
        tzs = ZONES if (s in TZ_TYPES or t in TZ_TYPES) else ["UTC"]
        for m in MODES:
            if "ok" not in analysis[m][f"{s} -> {t}"]:
                continue
            items = vals.get(s, [])
            if s == "string":
                items = [(lab, e) for lab, e in items if lab == "NULL" or string_kind_applies(kinds[lab], t)]
            for tz in tzs:
                work.append(((s, t, m, tz), pool.submit(run_unit, args.remote, m, tz, s, t, items)))
    for n, ((s, t, m, tz), f) in enumerate(work):
        for label, res in f.result().items():
            cases.setdefault((s, t, label, tz), {})[m] = res
        if n % 200 == 0:
            print(f"units {n}/{len(work)}", file=sys.stderr)

    # legacy flag cases
    flag_out = []
    fwork = []
    if not args.only:
        for name, extra, plan in flag_plan(vals):
            for s, t, items in plan:
                for m in ("ansi_on", "ansi_off"):
                    tzs = ["UTC", "America/Los_Angeles"] if (s in TZ_TYPES or t in TZ_TYPES) else ["UTC"]
                    for tz in tzs:
                        fwork.append(
                            (
                                (name, s, t, m, tz, extra),
                                pool.submit(run_unit_flag, args.remote, m, tz, s, t, items, extra),
                            )
                        )
    for (name, s, t, m, tz, _extra), f in fwork:
        for label, res in f.result().items():
            flag_out.append({"flag": name, "src": s, "tgt": t, "value": label, "tz": tz, "mode": m, **res})
    flag_types = {}
    if not args.only:
        for name, extra, plan in flag_plan(vals):
            for _src, t, _items in plan:
                if t not in flag_types.setdefault(name, {}):
                    flag_types[name][t] = analysis_one(
                        args.remote, mode_conf("ansi_on", "UTC", extra), "ansi_on", "void", t
                    )
    flag_out.sort(key=lambda c: (c["flag"], c["src"], c["tgt"], c["value"], c["tz"], c["mode"]))

    out_cases = [
        {"src": s, "tgt": t, "value": label, "tz": tz, **{m: res for m, res in sorted(r.items())}}
        for (s, t, label, tz), r in sorted(cases.items())
    ]
    doc = {
        "spark_version": version,
        "generator": "tests/scripts/gen_cast_matrix.py",
        "modes": {
            "ansi_on": "spark.sql.ansi.enabled=true, CAST",
            "ansi_off": "spark.sql.ansi.enabled=false, CAST",
            "try_cast": "spark.sql.ansi.enabled=true, TRY_CAST",
        },
        "flags": {name: extra for name, extra, _ in flag_plan(vals)} if not args.only else {},
        "sources": SOURCES,
        "targets": TARGETS,
        "analysis": analysis,
        "flag_result_types": flag_types,
        "cases": out_cases,
        "flag_cases": flag_out,
        "skipped_sources": sorted(skipped, key=lambda x: (x["src"], x["value"])),
    }
    json.dump(doc, sys.stdout, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    sys.stdout.write("\n")


def run_unit_flag(remote, mode, tz, src, tgt, items, extra):
    return run_unit(remote, mode, tz, src, tgt, items, extra)


if __name__ == "__main__":
    main()
