# SPDX-License-Identifier: Apache-2.0
"""Probe JSON decimal inference (`prefersDecimal`) in a real Spark run.

Spark's `JsonInferSchema` reads a JSON string as `new BigDecimal(s.replaceAll(",", ""))` first under
`prefersDecimal` (US locale; `DecimalType(precision, scale)`, any failure is no decimal), reads numbers past
`Long` and (with `prefersDecimal`) floating-point numbers as `DecimalType(max(p, s), s)`, and merges
decimals with `compatibleType` (the wider range plus the wider scale, `DOUBLE` past 38; an integral as
`DECIMAL(20, 0)` unless the decimal is wider). Each case is run through `schema_of_json` and as a one-line
JSON file read (the file's schema).

Usage (from repo root), once per version:

    v=4.3.0; out=tests/fixtures/spark-$v/json-prefers-decimal.json; mkdir -p "$(dirname "$out")"
    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:$v \\
        /opt/spark/bin/spark-submit --master 'local[1]' --driver-memory 1g /work/probe_json_prefers_decimal.py \\
        2>/dev/null > "$out"

Read by `crates/krishiv-session/tests/json_prefers_decimal.rs`.
"""

import json
import os
import sys
import tempfile

from pyspark.sql import SparkSession

PD = {"prefersDecimal": "true"}

STRINGS = [
    "1.5",
    "0.5",
    "0.01",
    "0.00",
    "0",
    "123",
    "-1.5",
    "+1.5",
    "1,000.5",
    "1,5",
    ",",
    "1e3",
    "1E-2",
    "1.5e1",
    "15e-1",
    ".5",
    "5.",
    "007.50",
    "-0.0",
    "12345678901234567890123456789012345678",
    "123456789012345678901234567890123456789",
    "1234567890123456789012345678901234567.8",
    " 1.5",
    "1.5 ",
    "NaN",
    "Infinity",
    "1.5d",
    "0x10",
    "1_0",
    "١٢",
    "1e2147483648",
    "2024-01-02",
]

CASES = [('{"t":' + json.dumps(s) + "}", PD) for s in STRINGS] + [
    ('{"t":"1.5"}', {}),
    ('{"t":"2024-01-02 03:04:05"}', {"prefersDecimal": "true", "inferTimestamp": "true"}),
    ('{"t":"1.5"}', {"prefersDecimal": "true", "inferTimestamp": "true"}),
    ('{"t":"1.5","n":2.5,"b":true}', {"prefersDecimal": "true", "primitivesAsString": "true"}),
    ('{"t":"1.000,5"}', {"prefersDecimal": "true", "locale": "de-DE"}),
    ('{"t":1.5}', PD),
    ('{"t":0.01}', PD),
    ('{"t":1e3}', PD),
    ('{"t":1.5e1}', PD),
    ('[{"t":1.5},{"t":12345.678}]', PD),
    ('[{"t":12345.6},{"t":1.234}]', PD),
    ('[{"t":"12345.6"},{"t":"1.234"}]', PD),
    ('[{"t":1},{"t":1.5}]', PD),
    ('[{"t":1.5},{"t":1}]', PD),
    ('[{"t":1},{"t":"1.5"}]', PD),
    ('[{"t":"1.5"},{"t":2.5}]', PD),
    ('[{"t":"1.5"},{"t":"abc"}]', PD),
    ('[{"t":"1.5"},{"t":true}]', PD),
    ('[{"t":"1.5"},{"t":null}]', PD),
    ('[{"t":"1.5"},{"t":""}]', PD),
    ('[{"t":"1.5"},{"t":2.5e300}]', {}),
    ('[{"t":1.5},{"t":"x"}]', PD),
    ('[{"t":9223372036854775808},{"t":1}]', {}),
    ('[{"t":1},{"t":9223372036854775808}]', {}),
    ('[{"t":12345678901234567890},{"t":1}]', {}),
    ('[{"t":123456789012345678901},{"t":1}]', {}),
    ('[{"t":9223372036854775808},{"t":1.5}]', {}),
    ('[{"t":9223372036854775808},{"t":1.5}]', PD),
    ('[{"t":"0.12345678901234567890123456789012345678"},{"t":"1234567890.5"}]', PD),
    ('[{"t":"0.1234567890123456789012345678901234567"},{"t":"1.5"}]', PD),
    ('[{"t":"12345678901234567890"},{"t":1}]', PD),
    ('[{"t":"1.5"},{"t":"12"},{"t":"-0.25"}]', PD),
    ('[["1.5", 2], ["3.25"]]', PD),
    # Non-numeric numbers (allowed by default): `getDecimalValue` fails on them, so `schema_of_json` raises
    # and a file read takes the record to the parse mode.
    ('{"t":NaN}', PD),
    ('{"t":Infinity}', PD),
    ('{"t":-Infinity}', PD),
    ('{"t":NaN}', {}),
    ('[{"t":NaN},{"t":1.5}]', PD),
    ('[{"t":1.5},{"t":-Infinity}]', PD),
    ('{"t":NaN}', {"prefersDecimal": "true", "mode": "FAILFAST"}),
    ('[{"t":NaN},{"t":1.5}]', {"prefersDecimal": "true", "mode": "DROPMALFORMED"}),
    ('[{"t":NaN},{"t":1.5}]', {"prefersDecimal": "true", "columnNameOfCorruptRecord": "bad"}),
]


def lit(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def error_of(e: Exception) -> str:
    # `getCondition` (4.0+) replaces `getErrorClass`, which warns on stdout.
    cond = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", lambda: None)
    got = cond()
    # A Py4J error (a file read's job failure): the first Spark condition, else the innermost Java class.
    j = getattr(e, "java_exception", None)
    while got is None and j is not None:
        name = j.getClass().getName()
        if name.startswith("org.apache.spark"):
            got = j.getCondition()
        elif j.getCause() is None:
            got = name.rsplit(".", 1)[-1]
        j = j.getCause()
    return "ERROR " + (got or type(e).__name__)


def main() -> None:
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    root = tempfile.mkdtemp()
    rows = []
    for i, (text, opts) in enumerate(CASES):
        pairs = ", ".join(lit(x) for kv in opts.items() for x in kv)
        arg = f", map({pairs})" if opts else ""
        try:
            got = spark.sql(f"SELECT schema_of_json({lit(text)}{arg}) AS s").collect()[0].s
        # The error class is the result.
        except Exception as e:
            got = error_of(e)
        path = os.path.join(root, f"case-{i}.json")
        with open(path, "w") as f:
            f.write(text + "\n")
        try:
            schema = spark.read.options(**opts).json(path).schema.simpleString()
        # The error class is the result.
        except Exception as e:
            schema = error_of(e)
        rows.append({"text": text, "options": opts, "schema_of_json": got, "file_schema": schema})
    out = {"spark": spark.version, "cases": rows}
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
