#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe the Spark-version differences of TIME in the CSV and JSON sources that Krishiv keys by profile.

Runs a REAL Spark (never from memory) in a throw-away container and prints one JSON object to stdout:
- `infer`: the schema CSV (`inferSchema`) and JSON (`inferTimestamp`) file reads and `schema_of_csv` infer for
  time-only texts, with `spark.sql.timeType.enabled` off and on. Spark 4.2 never infers TIME; 4.3.0 infers
  `time(6)` with the gate on (`krishiv.io.text.inferTime`, set from the profile).
- `read`: `12:34:56.123456789` read from a CSV file into `TIME(6)` with the gate on, under the default grammar and
  under `timeFormat=HH:mm:ss.SSSSSSSSS`, cast to STRING. Spark 4.2 cuts the default grammar to microseconds and
  keeps every digit a pattern reads (`krishiv.io.text.timeNanos`).
- `schema_of_json`: `schema_of_json(text, options)` over string values that read as times and timestamps, per
  setting (`spark.sql.timeType.enabled`, `spark.sql.timestampType`, `spark.sql.legacy.timeParserPolicy`): the
  `inferTimestamp` arm of `JsonInferSchema` (TIME only in 4.3.0, NTZ only by default type, the legacy fallback).
- `json_file`: the same texts as JSON files (one record per line), read with the same options and settings: the
  file read's inferred schema and the column `t` cast to STRING (a refusal's error class instead).
- `csv_file`: one-value CSV files (header `t`) under the default policy and `timeParserPolicy=LEGACY`, read three
  ways: `inferSchema`, schema `t DATE` and schema `t TIMESTAMP`; each gives the schema and `t` cast to STRING (a
  refusal's error class instead). LEGACY reads with `FastDateFormat` patterns and the lenient fallback.
- `csv_expr`: the `csv_file` cases as expressions under the same two policies: `schema_of_csv(text, options)` and
  `from_csv(text, 't DATE' | 't TIMESTAMP', options).t` cast to STRING (a refusal's error class instead).
- `json_read`: one-record JSON files read with schema `t DATE` and `t TIMESTAMP` under the same two policies.
  `JSONOptions.dateFormatInRead` is the option alone, so without `dateFormat` dates read leniently even with the
  fallback off (CSV reads `yyyy-MM-dd` strictly there, under `preferDate`).
- `write`: one row of a `TIMESTAMP`, a `DATE` and a `TIMESTAMP_NTZ` written as CSV and as JSON under the default
  policy and `timeParserPolicy=LEGACY` (in UTC and in `America/Los_Angeles`), per write option set; each gives the
  written line (a refusal's error class instead). LEGACY writes dates and timestamps with `FastDateFormat`
  (`yyyy-MM-dd'T'HH:mm:ss.SSSXXX` by default, Julian before 1582); `TIMESTAMP_NTZ` keeps the `java.time` pattern.

Usage (from repo root), once per version:

    v=4.2.1; out=tests/fixtures/spark-$v/time-text-profiles.json; mkdir -p "$(dirname "$out")"
    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:$v \\
        /opt/spark/bin/spark-submit --master 'local[1]' --driver-memory 1g /work/probe_time_text_profiles.py \\
        > "$out.tmp" && python3 -m json.tool "$out.tmp" > /dev/null && mv "$out.tmp" "$out"

Checked by `krishiv-session/tests/time_file_reads.rs` (`text_time_follows_the_profile_runs`) and
`krishiv-io/src/csv/time_tests.rs` (the 4.2 profile rows); the `schema_of_json` section by
`krishiv-session/tests/schema_of_json_time.rs`, the `json_file` section by `krishiv-session/tests/json_file_time.rs`,
the `csv_file`, `csv_expr` and `json_read` sections by `krishiv-session/tests/csv_file_legacy.rs`, the `write`
section by `krishiv-session/tests/text_write_legacy.rs`.
"""

import json
import logging
import os
import sys
import tempfile

from pyspark.sql import SparkSession

SOJ_SETTINGS = [
    ("default", {}),
    ("time", {"spark.sql.timeType.enabled": "true"}),
    ("ntz", {"spark.sql.timestampType": "TIMESTAMP_NTZ"}),
    ("time ntz", {"spark.sql.timeType.enabled": "true", "spark.sql.timestampType": "TIMESTAMP_NTZ"}),
    ("legacy", {"spark.sql.legacy.timeParserPolicy": "LEGACY"}),
]

ON = {"inferTimestamp": "true"}

SOJ_CASES = [
    ('{"t":"12:34:56"}', {}),
    ('{"t":"12:34:56"}', ON),
    ('{"t":"12:34:56.123456789"}', ON),
    ('{"t":"12:34"}', ON),
    ('{"t":"2024-01-02 03:04:05"}', {}),
    ('{"t":"2024-01-02 03:04:05"}', ON),
    ('{"t":"2024-01-02T03:04:05+01:00"}', ON),
    ('{"t":"2024-01-02"}', ON),
    ('{"t":"2024-1-2 3:4:5"}', ON),
    ('{"t":"not a time"}', ON),
    ('[{"t":"12:34:56"},{"t":"2024-01-02 03:04:05"}]', ON),
    ('[{"t":"2024-01-02 03:04:05"},{"t":"2024-01-02T03:04:05Z"}]', ON),
    ('[{"t":"2024-01-02 03:04:05"},{"t":"abc"}]', ON),
    ('[{"t":"12:34:56"},{"t":null}]', ON),
    ('{"t":"02/01/2024"}', {"inferTimestamp": "true", "timestampFormat": "dd/MM/yyyy"}),
    ('{"t":"2024-01-02 03:04:05"}', {"inferTimestamp": "true", "timestampFormat": "dd/MM/yyyy"}),
    ('{"t":"02/01/2024 03:04"}', {"inferTimestamp": "true", "timestampNTZFormat": "dd/MM/yyyy HH:mm"}),
    ('{"t":"12.34"}', {"inferTimestamp": "true", "timeFormat": "HH.mm"}),
    ('{"t":"12:34:56","n":1}', {"inferTimestamp": "true", "primitivesAsString": "true"}),
    ('{"t":"1.5"}', {"inferTimestamp": "true", "prefersDecimal": "true"}),
    ('{"t":"12:34:56"}', {"inferTimestamp": "true", "prefersDecimal": "true"}),
]

CSV_SETTINGS = [
    ("default", {}),
    ("legacy", {"spark.sql.legacy.timeParserPolicy": "LEGACY"}),
]

CSV_SCHEMAS = [("infer", None), ("date", "t DATE"), ("timestamp", "t TIMESTAMP")]

CSV_CASES = [
    ("2024-01-02", {}),
    ("2024-1-2", {}),
    ("2024-01-02 03:04:05", {}),
    ("2024-01-02T03:04:05", {}),
    ("2024-01-02T03:04:05.123+01:00", {}),
    ("2024-01-02T03:04:05.123456Z", {}),
    ("2024-01-02 03:04:05 GMT", {}),
    ("2024-01-02 03:04:05.1", {}),
    ("2024-1-2 3:4:5", {}),
    ("12:34:56", {}),
    ("not a date", {}),
    ("02/01/2024", {"dateFormat": "dd/MM/yyyy"}),
    ("02/01/2024 03:04", {"timestampFormat": "dd/MM/yyyy HH:mm"}),
    ("2024-01-02", {"dateFormat": "dd/MM/yyyy"}),
    ("2024-01-02 03:04:05", {"timestampFormat": "dd/MM/yyyy HH:mm"}),
    ("2024-01-02 03:04:05", {"timestampFormat": "dd/MM/yyyy HH:mm", "enableDateTimeParsingFallback": "false"}),
    ("2024-13-45", {}),
    ("2024-02-30", {"dateFormat": "yyyy-MM-dd"}),
    ("2024-01-02", {"preferDate": "true"}),
    ("2024-01-02T03:04:05.123+01:00", {"enableDateTimeParsingFallback": "true"}),
    ("2024-01-02 03:04:05", {"enableDateTimeParsingFallback": "true"}),
    # `dateFormatInRead` is `yyyy-MM-dd` only under `preferDate`: with it off the default reader stays lenient.
    ("2024-01-02 03:04:05", {"enableDateTimeParsingFallback": "false"}),
    ("2024-01-02 03:04:05", {"enableDateTimeParsingFallback": "false", "preferDate": "false"}),
    ("2024-1-2", {"enableDateTimeParsingFallback": "false"}),
    ("2024-1-2", {"enableDateTimeParsingFallback": "false", "preferDate": "false"}),
    ("2024-01-02 03:04:05", {"preferDate": "false"}),
]

JSON_READ_SCHEMAS = [("date", "t DATE"), ("timestamp", "t TIMESTAMP")]

JSON_READ_TEXTS = [
    "2024-01-02",
    "2024-1-2",
    "2024-01-02 03:04:05",
    "2024-01-02T03:04:05.123+01:00",
    "02/01/2024",
    "not a date",
]

JSON_READ_OPTIONS = [
    {},
    {"enableDateTimeParsingFallback": "false"},
    {"enableDateTimeParsingFallback": "true"},
    {"dateFormat": "dd/MM/yyyy"},
    {"dateFormat": "dd/MM/yyyy", "enableDateTimeParsingFallback": "false"},
    {"timestampFormat": "dd/MM/yyyy HH:mm", "enableDateTimeParsingFallback": "false"},
]

JSON_READ_CASES = [(t, o) for o in JSON_READ_OPTIONS for t in JSON_READ_TEXTS]

WRITE_SETTINGS = [
    ("default", {}),
    ("legacy", {"spark.sql.legacy.timeParserPolicy": "LEGACY"}),
    ("default la", {"spark.sql.session.timeZone": "America/Los_Angeles"}),
    ("legacy la", {"spark.sql.legacy.timeParserPolicy": "LEGACY", "spark.sql.session.timeZone": "America/Los_Angeles"}),
]

WRITE_ROWS = [
    ("2024-01-02 03:04:05.123456", "2024-01-02"),
    ("2024-01-02 03:04:05", "2024-07-02"),
    ("2024-07-02 13:04:05.5", "1999-12-31"),
    ("1500-03-01 00:00:00", "1500-03-01"),
    ("1582-10-14 12:00:00", "1582-10-14"),
    ("0001-01-01 00:00:00", "0001-01-01"),
    ("1883-11-18 12:00:00", "1883-11-18"),
]

WRITE_OPTIONS = [
    {},
    {"timestampFormat": "yyyy-MM-dd HH:mm:ss.SSSSSS"},
    {"timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSZ", "dateFormat": "dd/MM/yyyy"},
    {"timestampFormat": "dd MMM yyyy hh:mm a z", "dateFormat": "EEE, d MMM yyyy"},
    {"timestampFormat": "yyyy-MM-dd HH:mm:ss XXX", "timestampNTZFormat": "yyyy-MM-dd HH:mm"},
    {"dateFormat": "u yyyy"},
    {"timestampFormat": "yyyy-MM-dd b"},
    {"dateFormat": "yyyy-MM-dd b"},
]


def error_of(e: Exception) -> str:
    # `getCondition` (4.0+) replaces `getErrorClass`, which warns on stdout.
    cond = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", lambda: None)
    return "ERROR " + (cond() or type(e).__name__)


def lit(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def main() -> None:
    # PySpark 4's query-context logger writes failed queries to stdout, where the JSON goes.
    logging.getLogger("SQLQueryContextLogger").setLevel(logging.CRITICAL)
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    root = tempfile.mkdtemp()
    csv = os.path.join(root, "a.csv")
    with open(csv, "w") as f:
        f.write("id,t\n0,12:34:56\n1,01:02:03.5\n")
    js = os.path.join(root, "a.json")
    with open(js, "w") as f:
        f.write('{"id":0,"t":"12:34:56"}\n')
    nine = os.path.join(root, "nine.csv")
    with open(nine, "w") as f:
        f.write("t\n12:34:56.123456789\n")

    infer = {}
    for gate in ["false", "true"]:
        spark.conf.set("spark.sql.timeType.enabled", gate)
        c = spark.read.option("header", "true").option("inferSchema", "true").csv(csv)
        j = spark.read.option("inferTimestamp", "true").json(js)
        infer[gate] = {
            "csv": c.schema["t"].dataType.simpleString(),
            "json": j.schema["t"].dataType.simpleString(),
            "schema_of_csv": spark.sql("SELECT schema_of_csv('14:30:45') AS s").collect()[0].s,
        }

    spark.conf.set("spark.sql.timeType.enabled", "true")
    read = {}
    for name, opts in [("default", {}), ("HH:mm:ss.SSSSSSSSS", {"timeFormat": "HH:mm:ss.SSSSSSSSS"})]:
        df = spark.read.option("header", "true").options(**opts).schema("t TIME(6)").csv(nine)
        read[name] = df.selectExpr("CAST(t AS STRING) AS s").collect()[0].s

    spark.conf.set("spark.sql.timeType.enabled", "false")
    sojson = {}
    jfile = {}
    for setting, confs in SOJ_SETTINGS:
        for k, v in confs.items():
            spark.conf.set(k, v)
        rows = []
        for text, opts in SOJ_CASES:
            pairs = ", ".join(lit(x) for kv in opts.items() for x in kv)
            arg = f", map({pairs})" if opts else ""
            sql = f"SELECT schema_of_json({lit(text)}{arg}) AS s"
            try:
                got = spark.sql(sql).collect()[0].s
            # The error class is the result.
            except Exception as e:
                got = error_of(e)
            rows.append({"text": text, "options": opts, "schema": got})
        sojson[setting] = rows
        frows = []
        for i, (text, opts) in enumerate(SOJ_CASES):
            path = os.path.join(root, f"case-{setting.replace(' ', '-')}-{i}.json")
            with open(path, "w") as f:
                f.write(text + "\n")
            try:
                df = spark.read.options(**opts).json(path)
                schema = df.schema.simpleString()
                values = [r.s for r in df.selectExpr("CAST(t AS STRING) AS s").collect()]
            # The error class is the result.
            except Exception as e:
                schema = error_of(e)
                values = []
            frows.append({"text": text, "options": opts, "schema": schema, "values": values})
        jfile[setting] = frows
        for k in confs:
            spark.conf.unset(k)

    cfile = {}
    for setting, confs in CSV_SETTINGS:
        for k, v in confs.items():
            spark.conf.set(k, v)
        rows = []
        for i, (text, opts) in enumerate(CSV_CASES):
            path = os.path.join(root, f"csv-{setting}-{i}.csv")
            with open(path, "w") as f:
                f.write("t\n" + text + "\n")
            reads = {}
            for name, ddl in CSV_SCHEMAS:
                r = spark.read.option("header", "true").options(**opts)
                r = r.option("inferSchema", "true") if ddl is None else r.schema(ddl)
                try:
                    df = r.csv(path)
                    schema = df.schema.simpleString()
                    values = [x.s for x in df.selectExpr("CAST(t AS STRING) AS s").collect()]
                # The error class is the result.
                except Exception as e:
                    schema = error_of(e)
                    values = []
                reads[name] = {"schema": schema, "values": values}
            rows.append({"text": text, "options": opts, "reads": reads})
        cfile[setting] = rows
        for k in confs:
            spark.conf.unset(k)

    cexpr = {}
    for setting, confs in CSV_SETTINGS:
        for k, v in confs.items():
            spark.conf.set(k, v)
        rows = []
        for text, opts in CSV_CASES:
            # An empty `map()` is `MAP<VOID, VOID>`, which Spark refuses as options: no options, no argument.
            omap = "".join(f", {lit(k)}, {lit(v)}" for k, v in opts.items())
            omap = f", map({omap[2:]})" if omap else ""
            exprs = {"schema_of_csv": f"schema_of_csv({lit(text)}{omap})"}
            for name, ddl in CSV_SCHEMAS[1:]:
                exprs[name] = f"CAST(from_csv({lit(text)}, {lit(ddl)}{omap}).t AS STRING)"
            got = {}
            for name, e in exprs.items():
                try:
                    got[name] = {"value": spark.sql(f"SELECT {e} AS s").collect()[0].s}
                # The error class is the result.
                except Exception as ex:
                    got[name] = {"error": error_of(ex)}
            rows.append({"text": text, "options": opts, "exprs": got})
        cexpr[setting] = rows
        for k in confs:
            spark.conf.unset(k)

    jread = {}
    for setting, confs in CSV_SETTINGS:
        for k, v in confs.items():
            spark.conf.set(k, v)
        rows = []
        for i, (text, opts) in enumerate(JSON_READ_CASES):
            path = os.path.join(root, f"jread-{setting}-{i}.json")
            with open(path, "w") as f:
                f.write(json.dumps({"t": text}) + "\n")
            reads = {}
            for name, ddl in JSON_READ_SCHEMAS:
                try:
                    df = spark.read.options(**opts).schema(ddl).json(path)
                    schema = df.schema.simpleString()
                    values = [x.s for x in df.selectExpr("CAST(t AS STRING) AS s").collect()]
                # The error class is the result.
                except Exception as e:
                    schema = error_of(e)
                    values = []
                reads[name] = {"schema": schema, "values": values}
            rows.append({"text": text, "options": opts, "reads": reads})
        jread[setting] = rows
        for k in confs:
            spark.conf.unset(k)

    write = {}
    for setting, confs in WRITE_SETTINGS:
        for k, v in confs.items():
            spark.conf.set(k, v)
        rows = []
        for i, opts in enumerate(WRITE_OPTIONS):
            for j, (ts, d) in enumerate(WRITE_ROWS):
                df = spark.sql(f"SELECT TIMESTAMP {lit(ts)} AS ts, DATE {lit(d)} AS d, TIMESTAMP_NTZ {lit(ts)} AS ntz")
                lines = {}
                for fmt in ["csv", "json"]:
                    path = os.path.join(root, f"w-{setting.replace(' ', '-')}-{i}-{j}.{fmt}")
                    try:
                        df.coalesce(1).write.options(**opts).format(fmt).save(path)
                        parts = sorted(p for p in os.listdir(path) if p.startswith("part-"))
                        got = []
                        for p in parts:
                            with open(os.path.join(path, p)) as f:
                                got.extend(f.read().splitlines())
                    # The error class is the result.
                    except Exception as e:
                        got = [error_of(e)]
                    lines[fmt] = got
                rows.append({"timestamp": ts, "date": d, "options": opts, "lines": lines})
        write[setting] = rows
        for k in confs:
            spark.conf.unset(k)
        spark.conf.set("spark.sql.session.timeZone", "UTC")

    out = {
        "spark": spark.version,
        "infer": infer,
        "read": read,
        "schema_of_json": sojson,
        "json_file": jfile,
        "csv_file": cfile,
        "csv_expr": cexpr,
        "json_read": jread,
        "write": write,
    }
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    spark.stop()


if __name__ == "__main__":
    main()
