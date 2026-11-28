#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe how Spark reads the boolean options of its JSON, CSV and XML options, per value and entry point.

Runs a REAL Spark (never from memory) in a throw-away container and prints one JSON object to stdout:
`{"spark": version, "values": [...], "json": {option: {value: {entry: outcome}}}, "csv": ..., "xml": ...}`.
An outcome is `ok`, a Spark error class (`_LEGACY_ERROR_TEMP_2147` for `CSVOptions.getBool` / `XmlOptions.getBool`)
or a Java exception class (`java.lang.IllegalArgumentException` for Scala's `toBoolean`). Entries: the three
expressions of each format (`from_*`, `to_*`, `schema_of_*`), a file read of a one-record file (`read`, JSON and
CSV) and, for CSV, `from_csv` / `to_csv` / the read under `spark.sql.legacy.timeParserPolicy=LEGACY` (`*_legacy`, where
`preferDate` is never read). Values cover case, padding (never trimmed), non-booleans and `fal\\u017fe` (Java's
`equalsIgnoreCase` matches the long s to `s`; `toLowerCase` does not).
`single_quotes`: `{'a': 'x', 'b': 1}` with no options read as a JSON file (schema and rows), through `from_json` and
`schema_of_json`, and through `from_json` with `allowSingleQuotes=false`: `JSONOptions` defaults it to true.
`csv_values`: the results of `from_csv`, `to_csv` and `schema_of_csv` with `inferSchema`, `enforceSchema`,
`columnPruning` and `multiLine` unset, `true` and `false` (which of them change an expression's result).

Usage (from repo root), once per version:

    v=4.2.1; out=tests/fixtures/spark-$v/text-bool-options.json; mkdir -p "$(dirname "$out")"
    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:$v \\
        /opt/spark/bin/spark-submit --master 'local[1]' --driver-memory 1g /work/probe_text_bool_options.py \\
        > "$out.tmp" && python3 -m json.tool "$out.tmp" > /dev/null && mv "$out.tmp" "$out"
"""

import json
import os
import tempfile
import warnings

from pyspark.sql import SparkSession

warnings.filterwarnings("ignore")

VALUES = ["true", "TRUE", "True", " true", "true ", "yes", "1", "false", "FALSE", "", "fal\u017fe"]
JSON = [
    "enableDateTimeParsingFallback",
    "primitivesAsString",
    "prefersDecimal",
    "inferTimestamp",
    "allowComments",
    "allowSingleQuotes",
    "allowNumericLeadingZeros",
    "allowBackslashEscapingAnyCharacter",
    "allowUnquotedFieldNames",
    "allowUnquotedControlChars",
    "allowNonNumericNumbers",
    "ignoreNullFields",
    "dropFieldIfAllNull",
    "multiLine",
    "sortKeys",
    "pretty",
    "writeNonAsciiCharacterAsCodePoint",
    "useUnsafeRow",
]
CSV = [
    "header",
    "inferSchema",
    "ignoreLeadingWhiteSpace",
    "ignoreTrailingWhiteSpace",
    "preferDate",
    "enableDateTimeParsingFallback",
    "multiLine",
    "escapeQuotes",
    "quoteAll",
    "enforceSchema",
    "columnPruning",
]
XML = [
    "excludeAttribute",
    "ignoreSurroundingSpaces",
    "inferSchema",
    "ignoreNamespace",
    "prefersDecimal",
    "validateName",
    "preferDate",
    "multiLine",
    "useLegacyXMLParser",
]


def outcome(run):
    try:
        run()
        return "ok"
    except Exception as e:
        je = getattr(e, "java_exception", None)
        if je is None:
            cond = getattr(e, "getCondition", lambda: None)()
            return cond or "java.lang." + type(e).__name__
        # The Spark error, else the innermost IllegalArgumentException.
        while True:
            try:
                cond = je.getCondition()
            except Exception:
                cond = None
            if cond:
                return cond
            name = je.getClass().getName()
            if name.endswith("IllegalArgumentException") or je.getCause() is None:
                return name
            je = je.getCause()


def sql_lit(s):
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def main():
    spark = SparkSession.builder.getOrCreate()
    d = tempfile.mkdtemp()
    json_file = os.path.join(d, "a.json")
    with open(json_file, "w") as f:
        f.write('{"a": 1}\n')
    csv_file = os.path.join(d, "a.csv")
    with open(csv_file, "w") as f:
        f.write("a\n1\n")

    def q(text):
        return lambda: spark.sql(text).collect()

    def read(fmt, path, o, v):
        return lambda: spark.read.format(fmt).option(o, v).load(path).collect()

    def legacy(run):
        def go():
            spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
            try:
                run()
            finally:
                spark.conf.unset("spark.sql.legacy.timeParserPolicy")

        return go

    out = {"spark": spark.version, "values": VALUES, "json": {}, "csv": {}, "xml": {}}
    for o in JSON:
        out["json"][o] = {}
        for v in VALUES:
            m = f"map({sql_lit(o)}, {sql_lit(v)})"
            out["json"][o][v] = {
                "from_json": outcome(q(f"""select from_json('{{"a":1}}', 'a int', {m})""")),
                "to_json": outcome(q(f"select to_json(named_struct('a', 1, 'b', null), {m})")),
                "schema_of_json": outcome(q(f"""select schema_of_json('{{"a":1}}', {m})""")),
                "read": outcome(read("json", json_file, o, v)),
            }
    for o in CSV:
        out["csv"][o] = {}
        for v in VALUES:
            m = f"map({sql_lit(o)}, {sql_lit(v)})"
            from_csv = f"select from_csv('1', 'a int', {m})"
            to_csv = f"select to_csv(named_struct('a', 1), {m})"
            out["csv"][o][v] = {
                "from_csv": outcome(q(from_csv)),
                "to_csv": outcome(q(to_csv)),
                "schema_of_csv": outcome(q(f"select schema_of_csv('1', {m})")),
                "from_csv_legacy": outcome(legacy(q(from_csv))),
                "to_csv_legacy": outcome(legacy(q(to_csv))),
                "read": outcome(read("csv", csv_file, o, v)),
                "read_legacy": outcome(legacy(read("csv", csv_file, o, v))),
            }
    for o in XML:
        out["xml"][o] = {}
        for v in VALUES:
            m = f"map({sql_lit(o)}, {sql_lit(v)})"
            out["xml"][o][v] = {
                "from_xml": outcome(q(f"select from_xml('<r><a>1</a></r>', 'a int', {m})")),
                "to_xml": outcome(q(f"select to_xml(named_struct('a', 1), {m})")),
                "schema_of_xml": outcome(q(f"select schema_of_xml('<r><a>1</a></r>', {m})")),
            }
    sq_file = os.path.join(d, "sq.json")
    with open(sq_file, "w") as f:
        f.write("{'a': 'x', 'b': 1}\n")
    df = spark.read.json(sq_file)

    def first(text):
        return spark.sql(text).collect()[0][0]

    off = first("""select to_json(from_json("{'a': 'x'}", 'a string', map('allowSingleQuotes', 'false')))""")
    out["single_quotes"] = {
        "read_schema": df.schema.simpleString(),
        "read_rows": [[r.a, r.b] for r in df.collect()],
        "from_json": first("""select to_json(from_json("{'a': 'x'}", 'a string'))"""),
        "schema_of_json": first("""select schema_of_json("{'a': 'x'}")"""),
        "from_json_off": off,
    }
    out["csv_values"] = {}
    for o in ["inferSchema", "enforceSchema", "columnPruning", "multiLine"]:
        for v in [None, "true", "false"]:
            m = "" if v is None else f", map({sql_lit(o)}, {sql_lit(v)})"
            out["csv_values"][f"{o}={v}"] = {
                "from_csv": first(f"select to_json(from_csv('1,x', 'a int, b string'{m}))"),
                "to_csv": first(f"select to_csv(named_struct('a', 1, 'b', 'x'){m})"),
                "schema_of_csv": first(f"select schema_of_csv('1,x'{m})"),
            }
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
