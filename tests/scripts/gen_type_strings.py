#!/usr/bin/env python3
"""Generate the Spark DDL type-string fixture (P2-T001a, ADR 0009).

Every expectation comes from a REAL Spark 4.2 run (the `krishiv-bench` container), never from docs or
memory: each DDL string below is parsed by Spark's `CatalystSqlParser.parseDataType` inside the container
(`typestr/dump_types.scala`, run through `spark-shell`), and the fixture records the resulting type's
`simpleString`, `catalogString`, `sql`, `typeName` and `json`, or the error condition, SQLSTATE and
message parameters Spark raised. `DecimalType(p, s)` is also built directly for the precision/scale
bounds, and the error-condition entries of every error class seen are copied from Spark's
`error-conditions.json`.

The inputs cover every in-scope type (boolean, integral, float/double, string, binary, decimal, date,
timestamp, timestamp_ntz) with their aliases, case and whitespace variants, decimal bounds, a few
out-of-scope types (recorded so later tasks can reuse the fixture) and malformed strings. Output is
deterministic (no timestamps, stable ordering) and contains no local paths or hostnames.

Usage (from repo root; take the bench lock, start Spark, stop it afterwards; see bench/README.md):

    docker compose -f bench/baselines/docker-compose.yml up -d spark
    flock "$KRISHIV_DATA/.bench.lock" nice -n 19 \
      python3 tests/scripts/gen_type_strings.py --out tests/fixtures/spark-4.2.0/type-strings.json
    docker compose -f bench/baselines/docker-compose.yml stop spark
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gen_function_matrix import container_image, spark_shell  # noqa: E402

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_type_strings.py"

DDL = [
    # boolean and integral types, with aliases and case variants
    "boolean", "BOOLEAN", "Boolean",
    "tinyint", "byte", "TINYINT",
    "smallint", "short", "SMALLINT",
    "int", "integer", "INT", "Integer",
    "bigint", "long", "BIGINT",
    # floating point
    "float", "real", "FLOAT",
    "double", "DOUBLE", "double precision",
    # string and binary
    "string", "STRING", "binary", "BINARY",
    # decimal: defaults, aliases, spacing
    "decimal", "dec", "numeric", "DECIMAL", "Decimal",
    "decimal(10)", "decimal(10,2)", "decimal(10, 2)", "DECIMAL( 10 , 2 )", "dec(5,1)", "numeric(7,3)",
    "decimal(010,02)",
    # decimal bounds
    "decimal(1,0)", "decimal(1,1)", "decimal(18,0)", "decimal(18,18)", "decimal(19,0)", "decimal(38,0)",
    "DECIMAL(38, 38)", "decimal(38,10)", "decimal(0,0)", "decimal(0)",
    "decimal(39,0)", "DECIMAL(39, 0)", "decimal(39)", "decimal(100,0)", "decimal(5,6)", "decimal(38,39)",
    "decimal(10,-1)", "decimal(-1,0)", "decimal(99999999999,0)",
    # decimal syntax errors
    "decimal()", "decimal(10,2,3)", "decimal(10,2", "decimal(", "decimal(1.5,0)",
    # date and timestamps
    "date", "DATE",
    "timestamp", "TIMESTAMP", "timestamp_ltz", "TIMESTAMP_LTZ", "timestamp_ntz", "TIMESTAMP_NTZ",
    "TimeStamp_Ntz", "timestamp with local time zone", "timestamp without time zone",
    # whitespace
    "  int  ", "\tbigint\n", " decimal(10,2) ",
    # out of scope for P2-T001a (nested, interval, char/varchar, variant, null, time, collation)
    "void", "array<int>", "map<string,int>", "struct<a:int>", "char(5)", "varchar(10)",
    "interval year to month", "interval day to second", "variant", "time", "string collate utf8_lcase",
    # malformed
    "", "foo", "int,", "int int", "1", "`int`", "timestamp(3)",
    "timestamp with time zone", "timestamp_ntz(3)", "binary(5)", "decimal (10,2)", "string collate utf8_binary",
    # lexer: comments, Unicode whitespace and case folding, backquotes, number tokens
    "int /* c */", "int -- c", "/* c */ int", "int--c", "int /* a /* b */ c */", "int /* unclosed", "int\n-- c\n",
    "int\u00a0", "\u3000int", "int\u2028", "\u200bint", "\u0131nt", "\u017ftring", "\u00e9", "int\u00e9",
    "`a``b`", "`int", "``", "`int`(1)",
    ".5", "1.", "1e5", "10L", "1a", "decimal(10L,2)", "decimal(1e1,0)", "decimal(.5,0)",
    "decimal(5,99999999999)", "decimal(99999999999)",
]

DECIMALS = [(38, 38), (38, 0), (1, 0), (18, 18), (39, 0), (0, 0), (5, 6), (10, -1), (100, 0)]

LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="krishiv-bench-spark")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/type-strings.json")
    args = ap.parse_args()

    data = "".join(json.dumps({"ddl": d}) + "\n" for d in DDL)
    data += "".join(json.dumps({"precision": p, "scale": s}) + "\n" for p, s in DECIMALS)
    lines = spark_shell(args.container, (Path(__file__).parent / "typestr" / "dump_types.scala")
                        .read_text(encoding="utf-8"), data)
    version = next(line.split(" ", 1)[1] for line in lines if line.startswith("@@VERSION "))
    ddl = [json.loads(line[4:]) for line in lines if line.startswith("@@T ")]
    decimals = [json.loads(line[4:]) for line in lines if line.startswith("@@D ")]
    conditions = {}
    for line in lines:
        if line.startswith("@@EC "):
            ec = json.loads(line[5:])
            conditions[ec["name"]] = ec["entry"]
    if [c["ddl"] for c in ddl] != DDL or len(decimals) != len(DECIMALS):
        raise SystemExit("spark-shell output does not match the inputs")
    doc = {
        "format_version": FORMAT_VERSION,
        "spark_version": version,
        "image": container_image(args.container),
        "generator": GENERATOR,
        "notes": "DDL strings parsed by CatalystSqlParser.parseDataType; decimal_constructor builds "
                 "DecimalType(precision, scale) directly. Errors carry Spark's condition, SQLSTATE and "
                 "message parameters.",
        "ddl": ddl,
        "decimal_constructor": decimals,
        "error_conditions": dict(sorted(conditions.items())),
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if LOCAL_PATH.search(text):
        raise SystemExit("refusing to write a fixture that contains a local path")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {len(ddl)} DDL cases and {len(decimals)} decimal cases against Spark {version} -> {args.out}")


if __name__ == "__main__":
    main()
