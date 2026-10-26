#!/usr/bin/env python3
"""Generate Spark differential fixtures for the JSON reader (flat types).

Runs a REAL Spark (never from memory) in a throw-away container and records:

- `reads`: JSON-lines files read with `spark.read.json` under parse modes
  (`PERMISSIVE`, `DROPMALFORMED`, `FAILFAST`), with and without
  `_corrupt_record`, over the coercions this engine implements for flat
  types: boolean from strings (`"true"`/`"FALSE"` read, `"yes"`/`"1"` do
  not), boolean from numbers (`!= 0`, `NaN` reads `true`), `1.0` never as
  an integer, nested values as malformed records, and base64 binary;
- `infers`: `spark.read.json` schema inference over widening records.

Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[1]' /work/gen_json_goldens.py \
        > tests/fixtures/spark-4.2.0/json.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import base64
import json
import sys
import tempfile

from pyspark.sql import SparkSession


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def classes(e: Exception) -> list[str]:
    import re

    out = []
    for c in re.findall(r"\[([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\]", str(e)):
        if c not in out:
            out.append(c)
    return out or [type(e).__name__]


# (name, file text, read options, schema DDL or None to infer)
READS: list[tuple[str, str, dict[str, str], str | None]] = [
    (
        "boolean strings",
        '{"b":"true"}\n{"b":"TRUE"}\n{"b":"False"}\n{"b":"yes"}\n{"b":"1"}\n{"b":""}\n',
        {},
        "b boolean, _corrupt_record string",
    ),
    (
        "boolean numbers",
        '{"b":1}\n{"b":0}\n{"b":0.0}\n{"b":1.5}\n{"b":NaN}\n{"b":true}\n',
        {},
        "b boolean",
    ),
    (
        "nested is malformed",
        '{"a":1,"n":{"b":2}}\n{"a":2,"n":"x"}\n{"a":3,"n":[1]}\n',
        {},
        "a int, n string, _corrupt_record string",
    ),
    (
        "nested dropmalformed",
        '{"a":1,"n":{"b":2}}\n{"a":2,"n":"x"}\n',
        {"mode": "DROPMALFORMED"},
        "a int, n string",
    ),
    (
        "flat types",
        '{"b":true,"i":3,"f":1.5,"s":"a","bin":"qw==","dt":"2020-01-02"}\n'
        '{"b":"TRUE","i":1.0,"f":"NaN","s":3,"bin":null}\n',
        {},
        "b boolean, i int, f double, s string, bin binary, dt date, _corrupt_record string",
    ),
    (
        "binary bad base64",
        '{"bin":"qw=="}\n{"bin":"!!!"}\n',
        {},
        "bin binary, _corrupt_record string",
    ),
]

INFERS = [
    (
        "widening",
        '{"long":1,"dbl":1.5,"str":"x","b":true}\n'
        '{"long":2,"dbl":2,"str":3,"b":false}\n'
        '{"long":3.5,"dbl":null,"str":"y","b":1}\n',
        {},
    ),
]


def main() -> None:
    spark = (
        SparkSession.builder.appName("gen-json-goldens")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("OFF")
    out: dict = {
        "spark_version": spark.version,
        "generator": "tests/scripts/gen_json_goldens.py",
        "notes": (
            "JSON-lines reads and inference with flat schemas; each column "
            "cast to STRING in a UTC session, or the error classes outermost first."
        ),
        "reads": [],
        "infers": [],
    }
    for name, text, options, schema in READS:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            reader = spark.read.options(**options)
            df = reader.schema(schema).json(path) if schema else reader.json(path)
            try:
                cols = [c for c in df.columns]
                rows = df.selectExpr(*[f"cast({c} as string) as {c}" for c in cols]).collect()
                out["reads"].append(
                    {
                        "name": name,
                        "file": b64(text),
                        "options": options,
                        "schema": schema,
                        "rows": [[r[c] for c in cols] for r in rows],
                    }
                )
            except Exception as e:
                out["reads"].append(
                    {
                        "name": name,
                        "file": b64(text),
                        "options": options,
                        "schema": schema,
                        "error": classes(e),
                    }
                )
        finally:
            import os

            os.unlink(path)
    for name, text, options in INFERS:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            try:
                df = spark.read.options(**options).json(path)
                out["infers"].append(
                    {"name": name, "file": b64(text), "options": options, "inferred": df.schema.simpleString()}
                )
            except Exception as e:
                out["infers"].append({"name": name, "file": b64(text), "options": options, "error": classes(e)})
        finally:
            import os

            os.unlink(path)
    spark.stop()
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
