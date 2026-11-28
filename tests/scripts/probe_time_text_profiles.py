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

Usage (from repo root), once per version:

    v=4.2.1; out=tests/fixtures/spark-$v/time-text-profiles.json; mkdir -p "$(dirname "$out")"
    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:$v \\
        /opt/spark/bin/spark-submit --master 'local[1]' --driver-memory 1g /work/probe_time_text_profiles.py \\
        > "$out.tmp" && python3 -m json.tool "$out.tmp" > /dev/null && mv "$out.tmp" "$out"

Checked by `krishiv-session/tests/time_file_reads.rs` (`text_time_follows_the_profile_runs`) and
`krishiv-io/src/csv/time_tests.rs` (the 4.2 profile rows); the `schema_of_json` section by
`krishiv-session/tests/schema_of_json_time.rs`, the `json_file` section by `krishiv-session/tests/json_file_time.rs`,
the `csv_file` section by `krishiv-session/tests/csv_file_legacy.rs`.
"""

import json
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
]


def lit(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def main() -> None:
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
                got = "ERROR " + (getattr(e, "getErrorClass", lambda: None)() or type(e).__name__)
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
                schema = "ERROR " + (getattr(e, "getErrorClass", lambda: None)() or type(e).__name__)
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
                    schema = "ERROR " + (getattr(e, "getErrorClass", lambda: None)() or type(e).__name__)
                    values = []
                reads[name] = {"schema": schema, "values": values}
            rows.append({"text": text, "options": opts, "reads": reads})
        cfile[setting] = rows
        for k in confs:
            spark.conf.unset(k)

    out = {
        "spark": spark.version,
        "infer": infer,
        "read": read,
        "schema_of_json": sojson,
        "json_file": jfile,
        "csv_file": cfile,
    }
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    spark.stop()


if __name__ == "__main__":
    main()
