#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Spark differential fixtures for the JSON reader's nested types.

Runs a REAL Spark (never from memory) in a throw-away container and records,
for JSON-lines (and `multiLine`) files read with `spark.read.json`:

- `infers`: the inferred schema (`StructType.simpleString`) of nested objects
  and arrays: field order, struct and array merging, conflicts, empty
  objects and arrays, `dropFieldIfAllNull`, `primitivesAsString`,
  `prefersDecimal`, top-level arrays;
- `reads`: the rows read with the inferred schema or a DDL schema (structs,
  arrays, maps), under the three parse modes, with partial results for
  values that do not fit the schema and `_corrupt_record`.

Each row is a list of column values as JSON: a struct is an object with its
fields in schema order, an array a list, a map an object, a decimal a
string, binary base64, NaN and infinities their names as strings; anything
else as `json.dumps` writes the Python value.
An error is recorded as its error classes, outermost first.

Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[1]' /work/gen_json_nested_goldens.py \
        > tests/fixtures/spark-4.2.0/json-nested.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import base64
import datetime
import decimal
import json
import os
import re
import sys
import tempfile

from pyspark.sql import Row, SparkSession


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def classes(e: Exception) -> list[str]:
    out = []
    for c in re.findall(r"\[([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\]", str(e)):
        if c not in out:
            out.append(c)
    return out or [type(e).__name__]


def plain(v):
    """A collected value as JSON."""
    if isinstance(v, Row):
        return {k: plain(x) for k, x in zip(v.__fields__, v, strict=True)}
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [plain(x) for x in v]
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return base64.b64encode(bytes(v)).decode("ascii")
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    if isinstance(v, float) and v != v:
        return "NaN"
    if isinstance(v, float) and v in (float("inf"), float("-inf")):
        return "Infinity" if v > 0 else "-Infinity"
    return v


GH = (
    '{"id":"1","type":"PushEvent","actor":{"id":7,"login":"a","url":"u"},'
    '"repo":{"id":9,"name":"r/x"},"payload":{"size":1,"commits":[{"sha":"s1","distinct":true}]},'
    '"public":true,"created_at":"2024-01-15T12:00:00Z"}\n'
    '{"id":"2","type":"CreateEvent","actor":{"id":8,"login":"b","url":"v","display_login":"B"},'
    '"repo":{"id":10,"name":"r/y"},"payload":{"ref":"main","ref_type":"branch","pusher_type":"user"},'
    '"public":true,"created_at":"2024-01-15T12:00:01Z","org":{"id":3,"login":"o"}}\n'
    '{"id":"3","type":"PushEvent","actor":{"id":7,"login":"a","url":"u"},'
    '"repo":{"id":9,"name":"r/x"},"payload":{"size":2,"commits":[{"sha":"s2","distinct":false},'
    '{"sha":"s3","distinct":true,"message":"m"}]},"public":true,"created_at":"2024-01-15T12:00:02Z"}\n'
)

# (name, file text, read options)
INFERS: list[tuple[str, str, dict[str, str]]] = [
    ("root order", '{"b":1,"a":2,"C":3,"_x":4}\n{"aa":5}\n', {}),
    ("nested order", '{"s":{"z":1,"y":"t","B":true}}\n', {}),
    ("nested case duplicates", '{"s":{"z":1,"y":"t","Y":true}}\n', {}),
    ("malformed line", 'nope\n{"a":1}\n', {}),
    ("malformed line dropmalformed", 'nope\n{"a":1}\n', {"mode": "DROPMALFORMED"}),
    ("malformed line failfast", 'nope\n{"a":1}\n', {"mode": "FAILFAST"}),
    ("top-level scalar failfast", '1\n{"a":2}\n', {"mode": "FAILFAST"}),
    ("top-level scalar dropmalformed", '1\n{"a":2}\n', {"mode": "DROPMALFORMED"}),
    ("decimal strings", '{"a":"1.25","b":"x","c":"1e3","d":"-0.5"}\n', {"prefersDecimal": "true"}),
    ("null with primitives as string", '{"a":null,"b":1}\n', {"primitivesAsString": "true"}),
    ("unicode name order", '{"\u00e9":1,"z":2,"\U0001f600":3,"\uff21":4}\n', {}),
    ("depth 1000", '{"a":' + "[" * 999 + "]" * 999 + "}\n", {}),
    ("depth 1001", '{"a":' + "[" * 1000 + "]" * 1000 + "}\n", {}),
    ("struct merge", '{"s":{"a":1}}\n{"s":{"b":"x"}}\n{"s":{"a":2.5,"c":null}}\n', {}),
    ("array widen", '{"a":[1,2]}\n{"a":[1.5]}\n{"a":[]}\n{"a":null}\n', {}),
    ("array mixed elements", '{"a":[1,"x",true]}\n', {}),
    ("array of structs", '{"a":[{"x":1},{"y":"t"}]}\n{"a":[{"x":2,"z":[true]}]}\n', {}),
    ("array of arrays", '{"a":[[1],[2,3]]}\n{"a":[[],[null]]}\n', {}),
    ("empty array only", '{"a":[]}\n{"a":[]}\n', {}),
    ("array of nulls", '{"a":[null,null]}\n', {}),
    ("empty object dropped", '{"e":{},"k":1}\n', {}),
    ("empty object in array", '{"e":[{}],"k":1}\n', {}),
    ("struct with empty name", '{"s":{"":1,"a":2}}\n', {}),
    ("struct vs scalar", '{"s":{"a":1}}\n{"s":5}\n', {}),
    ("array vs struct", '{"s":[1]}\n{"s":{"a":1}}\n', {}),
    ("array vs scalar", '{"s":[1]}\n{"s":"x"}\n', {}),
    ("null then struct", '{"s":null}\n{"s":{"a":1}}\n', {}),
    ("empty string field", '{"s":""}\n{"s":1}\n', {}),
    ("empty string only", '{"s":""}\n', {}),
    ("empty string in struct", '{"s":{"a":""}}\n{"s":{"a":2}}\n', {}),
    ("decimal in array", '{"a":[1.25,10.5]}\n', {"prefersDecimal": "true"}),
    ("primitives as string", '{"s":{"a":1,"b":true},"a":[1.5,2]}\n', {"primitivesAsString": "true"}),
    ("drop all-null nested", '{"s":{"a":null,"b":1},"n":null,"e":[]}\n', {"dropFieldIfAllNull": "true"}),
    ("drop all-null struct", '{"s":{"a":null},"k":1}\n', {"dropFieldIfAllNull": "true"}),
    ("deep", '{"a":{"b":{"c":{"d":[{"e":[1]}]}}}}\n', {}),
    ("top-level array", '[{"a":1},{"b":"x"}]\n{"a":2}\n', {}),
    ("top-level empty array", '[]\n{"a":2}\n', {}),
    ("top-level scalar", '1\n{"a":2}\n', {}),
    ("top-level array of scalars", '[1,2]\n{"a":2}\n', {}),
    ("multiline nested", '[\n {"a": {"b": [1, 2]}},\n {"a": {"c": "x"}}\n]\n', {"multiLine": "true"}),
    ("github archive sample", GH, {}),
    ("root duplicate keys", '{"a":1,"a":2}\n', {}),
    ("nested duplicate keys", '{"s":{"a":1,"a":"x"}}\n', {}),
]

# (name, file text, read options, schema DDL or None to infer)
READS: list[tuple[str, str, dict[str, str], str | None]] = [
    ("inferred nested", '{"s":{"b":"x","a":1},"arr":[1,2,null],"k":1}\n{"s":null,"arr":[],"k":2}\n{"k":3}\n', {}, None),
    ("inferred array of structs", '{"a":[{"x":1},{"y":"t"}]}\n{"a":[{"x":2,"z":[true]},null]}\n', {}, None),
    ("inferred conflict as text", '{"s":{"a": 1, "b":[1, 2]}}\n{"s":5}\n{"s":[ true ]}\n', {}, None),
    ("inferred top-level array", '[{"a":1},{"b":"x"}]\n{"a":2}\n[]\n', {}, None),
    ("inferred github archive", GH, {}, None),
    ("inferred multiline", '[\n {"a": {"b": [1, 2]}},\n {"a": {"c": "x"}}\n]\n', {"multiLine": "true"}, None),
    (
        "struct partial",
        '{"s":{"a":1,"b":"x"},"k":1}\n{"s":{"a":"bad","b":"y"},"k":2}\n{"s":7,"k":3}\n{"s":"","k":4}\n',
        {},
        "s struct<a:int,b:string>, k int, _corrupt_record string",
    ),
    (
        "struct partial dropmalformed",
        '{"s":{"a":1,"b":"x"},"k":1}\n{"s":{"a":"bad","b":"y"},"k":2}\n',
        {"mode": "DROPMALFORMED"},
        "s struct<a:int,b:string>, k int",
    ),
    (
        "struct partial failfast",
        '{"s":{"a":1,"b":"x"},"k":1}\n{"s":{"a":"bad","b":"y"},"k":2}\n',
        {"mode": "FAILFAST"},
        "s struct<a:int,b:string>, k int",
    ),
    (
        "array elements",
        '{"a":[1,null,3],"k":1}\n{"a":[1,"x",3],"k":2}\n{"a":5,"k":3}\n{"a":[],"k":4}\n{"a":[1.5],"k":5}\n',
        {},
        "a array<int>, k int, _corrupt_record string",
    ),
    (
        "array of structs partial",
        '{"a":[{"x":1},{"x":"bad"},null,{"y":2}],"k":1}\n',
        {},
        "a array<struct<x:int,y:int>>, k int, _corrupt_record string",
    ),
    (
        "nested arrays",
        '{"a":[[1,2],[],null,[3]]}\n{"a":[[1],["x"]]}\n',
        {},
        "a array<array<int>>, _corrupt_record string",
    ),
    (
        "map",
        '{"m":{"x":1,"y":null,"z":3}}\n{"m":{}}\n{"m":{"x":"bad","y":2}}\n{"m":[1]}\n{"m":null}\n',
        {},
        "m map<string,int>, _corrupt_record string",
    ),
    ("map of structs", '{"m":{"x":{"a":1},"y":{"a":2,"b":3}}}\n', {}, "m map<string,struct<a:int>>"),
    (
        "struct missing and extra fields",
        '{"s":{"b":"x","extra":{"q":1},"a":2}}\n{"s":{}}\n',
        {},
        "s struct<a:int,b:string,c:double>",
    ),
    (
        "struct duplicate keys",
        '{"s":{"a":1,"a":2}}\n',
        {},
        "s struct<a:int>, _corrupt_record string",
    ),
    (
        "string column holds nested text",
        '{"s":{"a": 1, "b": [1, 2]},"t":[ 1 , "x" ]}\n',
        {},
        "s string, t string",
    ),
    (
        "string column holds nested text multiline",
        '{"s":{"a": 1, "b": [1, 2]},\n "t":[ 1 , "x" ]}\n',
        {"multiLine": "true"},
        "s string, t string",
    ),
    (
        "struct case sensitivity",
        '{"s":{"A":1,"b":2}}\n',
        {},
        "s struct<a:int,B:int>",
    ),
    (
        "struct of scalar types",
        '{"s":{"b":"true","d":"2020-01-02","ts":"2020-01-02T03:04:05Z","dec":"1.25","bin":"qw=="}}\n',
        {},
        "s struct<b:boolean,d:date,ts:timestamp,dec:decimal(5,2),bin:binary>, _corrupt_record string",
    ),
    (
        "array root schema mismatch",
        '{"a":{"x":1}}\n',
        {},
        "a array<struct<x:int>>, _corrupt_record string",
    ),
    (
        "depth 1000 read",
        '{"a":' + "[" * 999 + "]" * 999 + ',"k":1}\n',
        {},
        "a string, k int, _corrupt_record string",
    ),
    (
        "depth 1001 read",
        '{"a":' + "[" * 1000 + "]" * 1000 + ',"k":1}\n',
        {},
        "a string, k int, _corrupt_record string",
    ),
    (
        "deep nesting",
        '{"a":' + "[" * 1200 + "]" * 1200 + ',"k":1}\n',
        {},
        "a string, k int, _corrupt_record string",
    ),
]


MODES = ["PERMISSIVE", "DROPMALFORMED", "FAILFAST"]

# Top-level values that are not one object, read in each parse mode.
for _name, _text in [
    ("root array of scalars", "[1,2]\n"),
    ("root array of null", "[null]\n"),
    ("root array with a scalar", '[{"a":1},2]\n'),
    ("root array partial", '[{"a":"bad"},{"a":1}]\n'),
    ("root scalar", "1\n"),
    ("root null", "null\n"),
]:
    for _mode in MODES:
        READS.append((f"{_name} {_mode}", _text, {"mode": _mode}, "a int, _corrupt_record string"))
READS += [
    (
        "root array partial renamed corrupt column",
        '[{"a":"bad"},{"a":1}]\n',
        {"columnNameOfCorruptRecord": "bad"},
        "a int, bad string",
    ),
    ("root array string as double failfast", '[{"d":"abc"}]\n', {"mode": "FAILFAST"}, "d double"),
    ("string as double failfast", '{"d":"abc"}\n', {"mode": "FAILFAST"}, "d double"),
    ("nested string as double failfast", '{"s":{"d":"abc"}}\n', {"mode": "FAILFAST"}, "s struct<d:double>"),
    ("empty string as int failfast", '{"i":""}\n', {"mode": "FAILFAST"}, "i int"),
    ("empty string in struct failfast", '{"s":{"i":""}}\n', {"mode": "FAILFAST"}, "s struct<i:int>"),
    (
        "non-numeric numbers",
        '{"d":NaN}\n{"d":"NaN"}\n{"d":"-Infinity"}\n{"d":"+INF"}\n{"d":Infinity}\n',
        {},
        "d double, _corrupt_record string",
    ),
    (
        "non-numeric numbers off",
        '{"d":NaN}\n{"d":"NaN"}\n{"d":"-Infinity"}\n{"d":Infinity}\n{"d":1.5}\n',
        {"allowNonNumericNumbers": "false"},
        "d double, _corrupt_record string",
    ),
]

# Reads whose values the row encoding cannot show (duplicate map keys, a
# wrapped timestamp): the listed expressions over the read instead.
SELECTS: list[tuple[str, str, dict[str, str], str, list[str]]] = [
    ("map duplicate keys", '{"m":{"x":1,"x":2,"y":3}}\n', {}, "m map<string,int>", ["size(m)", "map_keys(m)"]),
    (
        "integral timestamp overflow",
        '{"ts":100000000000000}\n{"ts":-5}\n',
        {},
        "ts timestamp, _corrupt_record string",
        ["unix_micros(ts)", "_corrupt_record"],
    ),
]


def run(spark, text: str, options: dict[str, str], schema: str | None):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        reader = spark.read.options(**options)
        df = reader.schema(schema).json(path) if schema else reader.json(path)
        inferred = df.schema.simpleString()
        rows = [[plain(r[c]) for c in range(len(df.columns))] for r in df.collect()]
        return {"inferred": inferred, "rows": rows}
    except Exception as e:
        return {"error": classes(e)}
    finally:
        os.unlink(path)


def main() -> None:
    spark = (
        SparkSession.builder.appName("gen-json-nested-goldens")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    out: dict = {
        "spark_version": spark.version,
        "generator": "tests/scripts/gen_json_nested_goldens.py",
        "notes": (
            "JSON reads with nested types: inferred schemas (simpleString) and rows as JSON "
            "(struct = object in schema order, map = object, decimal = string, binary = base64), "
            "or the error classes outermost first. UTC session, ANSI off."
        ),
        "infers": [],
        "reads": [],
    }
    for name, text, options in INFERS:
        r = run(spark, text, options, None)
        r.pop("rows", None)
        out["infers"].append({"name": name, "file": b64(text), "options": options, **r})
    for name, text, options, schema in READS:
        r = run(spark, text, options, schema)
        out["reads"].append({"name": name, "file": b64(text), "options": options, "schema": schema, **r})
    out["selects"] = []
    for name, text, options, schema, exprs in SELECTS:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            df = spark.read.options(**options).schema(schema).json(path).selectExpr(*exprs)
            rows = [[plain(r[c]) for c in range(len(exprs))] for r in df.collect()]
            out["selects"].append(
                {"name": name, "file": b64(text), "options": options, "schema": schema, "exprs": exprs, "rows": rows}
            )
        finally:
            os.unlink(path)
    spark.stop()
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
