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
timestamp, timestamp_ntz, time, geometry, geography) with their aliases, case and whitespace variants,
decimal bounds, a few out-of-scope types (recorded so later tasks can reuse the fixture) and malformed
strings. Output is deterministic (no timestamps, stable ordering) and contains no local paths or hostnames.

Usage (from repo root; take the bench lock, start Spark, stop it afterwards; see bench/README.md):

    docker compose -f bench/baselines/docker-compose.yml up -d spark
    flock "$KRISHIV_DATA/.bench.lock" nice -n 19 \
      python3 tests/scripts/gen_type_strings.py --out tests/fixtures/spark-4.2.0/type-strings.json
    docker compose -f bench/baselines/docker-compose.yml stop spark

`--spark-home DIR` runs the same Scala script with `DIR/bin/spark-shell` instead of the container, for
example with the Spark that a `pyspark==4.2.0` install ships (`DIR` = its `pyspark` package directory).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gen_function_matrix import container_image, spark_shell

FORMAT_VERSION = 1
GENERATOR = "tests/scripts/gen_type_strings.py"

DDL = [
    # boolean and integral types, with aliases and case variants
    "boolean",
    "BOOLEAN",
    "Boolean",
    "tinyint",
    "byte",
    "TINYINT",
    "smallint",
    "short",
    "SMALLINT",
    "int",
    "integer",
    "INT",
    "Integer",
    "bigint",
    "long",
    "BIGINT",
    # floating point
    "float",
    "real",
    "FLOAT",
    "double",
    "DOUBLE",
    "double precision",
    # string and binary
    "string",
    "STRING",
    "binary",
    "BINARY",
    # decimal: defaults, aliases, spacing
    "decimal",
    "dec",
    "numeric",
    "DECIMAL",
    "Decimal",
    "decimal(10)",
    "decimal(10,2)",
    "decimal(10, 2)",
    "DECIMAL( 10 , 2 )",
    "dec(5,1)",
    "numeric(7,3)",
    "decimal(010,02)",
    # decimal bounds
    "decimal(1,0)",
    "decimal(1,1)",
    "decimal(18,0)",
    "decimal(18,18)",
    "decimal(19,0)",
    "decimal(38,0)",
    "DECIMAL(38, 38)",
    "decimal(38,10)",
    "decimal(0,0)",
    "decimal(0)",
    "decimal(39,0)",
    "DECIMAL(39, 0)",
    "decimal(39)",
    "decimal(100,0)",
    "decimal(5,6)",
    "decimal(38,39)",
    "decimal(10,-1)",
    "decimal(-1,0)",
    "decimal(99999999999,0)",
    # decimal syntax errors
    "decimal()",
    "decimal(10,2,3)",
    "decimal(10,2",
    "decimal(",
    "decimal(1.5,0)",
    # date and timestamps
    "date",
    "DATE",
    "timestamp",
    "TIMESTAMP",
    "timestamp_ltz",
    "TIMESTAMP_LTZ",
    "timestamp_ntz",
    "TIMESTAMP_NTZ",
    "TimeStamp_Ntz",
    "timestamp with local time zone",
    "timestamp without time zone",
    # whitespace
    "  int  ",
    "\tbigint\n",
    " decimal(10,2) ",
    # out of scope for P2-T001a (nested, interval, char/varchar, variant, null, time, collation)
    "void",
    "array<int>",
    "map<string,int>",
    "struct<a:int>",
    "char(5)",
    "varchar(10)",
    "interval year to month",
    "interval day to second",
    "variant",
    "time",
    "string collate utf8_lcase",
    # time (P2-T181): precision bounds and the multi-word forms
    "TIME",
    "time(0)",
    "time(3)",
    "TIME(6)",
    "time(06)",
    "time(7)",
    "time(-1)",
    "time()",
    "time(1,2)",
    "time(99999999999)",
    "time without time zone",
    "TIME(3) WITHOUT TIME ZONE",
    "time with time zone",
    "time with local time zone",
    "time(3) with time zone",
    "time_ntz",
    "time without",
    "time(3) without time",
    # geometry and geography (P2-T181): SRIDs, ANY, malformed parameters
    "geometry(0)",
    "GEOMETRY(4326)",
    "geometry(3857)",
    "geometry( 3857 )",
    "geometry(04326)",
    "geometry(any)",
    "GEOMETRY(ANY)",
    "geometry(Any)",
    "geometry(`any`)",
    "geometry(anything)",
    "geometry",
    "geometry()",
    "geometry(9999)",
    "geometry(-1)",
    "geometry(1)",
    "geometry(99999999999)",
    "geometry(4326,0)",
    "geometry(srid:0)",
    "geometry(any",
    "geometry(any,1)",
    "geography(any) int",
    "geography(4326)",
    "GEOGRAPHY(ANY)",
    "geography(any)",
    "geography",
    "geography(0)",
    "geography(3857)",
    "geography(9999)",
    "geography(-1)",
    "geography(4326, spherical)",
    # malformed
    "",
    "foo",
    "int,",
    "int int",
    "1",
    "`int`",
    "timestamp(3)",
    "timestamp with time zone",
    "timestamp_ntz(3)",
    "binary(5)",
    "decimal (10,2)",
    "string collate utf8_binary",
    # lexer: comments, Unicode whitespace and case folding, backquotes, number tokens
    "int /* c */",
    "int -- c",
    "/* c */ int",
    "int--c",
    "int /* a /* b */ c */",
    "int /* unclosed",
    "int\n-- c\n",
    "int\u00a0",
    "\u3000int",
    "int\u2028",
    "int\u2029",
    "\u200bint",
    "\u0131nt",
    "\u017ftring",
    "\u00e9",
    "int\u00e9",
    "`a``b`",
    "`int",
    "``",
    "`int`(1)",
    ".5",
    "1.",
    "1e5",
    "10L",
    "1a",
    "decimal(10L,2)",
    "decimal(1e1,0)",
    "decimal(.5,0)",
    "decimal(5,99999999999)",
    "decimal(99999999999)",
    # nested types (P2-T001b): arrays, maps, structs, nesting, nullability, comments, quoting
    "ARRAY<INT>",
    "array< int >",
    "Array<String>",
    "array<array<int>>",
    "array<array<array<array<int>>>>",
    "array<decimal(10,2)>",
    "array<void>",
    "array<timestamp_ntz>",
    "array<decimal(0,0)>",
    "MAP<STRING, ARRAY<DOUBLE>>",
    "map<int,map<int,int>>",
    "map<map<int,int>,int>",
    "map<array<int>,int>",
    "map<struct<a:int>,string>",
    "map<void,int>",
    "map< string , int >",
    "struct<a int>",
    "struct<a:int,b:string>",
    "STRUCT<a: INT, b: STRING>",
    "struct<A:int,b:STRING>",
    "struct<>",
    "struct< >",
    "STRUCT<>",
    "struct<a:int not null>",
    "struct<a:int NOT NULL,b:string>",
    "struct<a:int comment 'hello'>",
    "struct<a:int not null comment 'x y'>",
    'struct<a:int comment "dq">',
    "struct<a:int comment 'it\\'s'>",
    "struct<a:int comment 'a\\\\b'>",
    "struct<a:int comment ''>",
    "struct<a:int comment 'tab\\there'>",
    "struct<a:int comment 'q\"d'>",
    "struct<`a b`:int>",
    "struct<`a``b`:int>",
    "struct<`a.b`:int>",
    "struct<`1a`:int>",
    "struct<`1`:int>",
    "struct<_a:int>",
    "struct<a1:int>",
    "struct<select:int>",
    "struct<`select`:int>",
    "struct<from:int>",
    "struct<a:int,a:string>",
    "struct<é:int>",
    "struct<`é`:int>",
    "struct<`a:b`:int>",
    "struct<` `:int>",
    "struct<a:array<struct<b:map<string,array<double>>>>>",
    "ARRAY<STRUCT<a: INT, b: MAP<STRING, ARRAY<DOUBLE>>>>",
    "map<string,struct<a:array<map<int,struct<b:int>>>>>",
    "struct<a:struct<b:struct<c:struct<d:int>>>>",
    "array<struct<a:int not null>>",
    "struct<a:decimal(10,2),b:timestamp_ntz,c:binary,d:date>",
    "struct<a:array<int>>",
    "struct<c0:int,c1:int,c2:int,c3:int,c4:int,c5:int,c6:int,c7:int,c8:int,c9:int,c10:int,c11:int,c12:int,c13:int,c14:int,c15:int,c16:int,c17:int,c18:int,c19:int,c20:int,c21:int,c22:int,c23:int,c24:int,c25:int,c26:int,c27:int,c28:int,c29:int>",
    "array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<array<int>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
    "array",
    "array<>",
    "array<int",
    "array<int,int>",
    "array(int)",
    "array<foo>",
    "array<int>>",
    "map",
    "map<int>",
    "map<int,int,int>",
    "map<int,>",
    "map<,int>",
    "struct",
    "struct<a>",
    "struct<a:>",
    "struct<a:int,>",
    "struct<,a:int>",
    "struct<a-b:int>",
    "struct<a:int null>",
    "struct<a:int not>",
    "struct<a:int comment>",
    "struct<a:int comment 5>",
    "struct<a:int comment 'x' not null>",
    "array<int> not null",
    "struct<a:int>>",
    "struct<a:foo>",
    "struct<a::int>",
    "array<int not null>",
    "struct<a:decimal(39,0)>",
    "array<decimal(5,6)>",
    "struct<a:int comment 'c\\u0001\\u0085\\u2028\\u00e9\\u007f\\u001f'>",
    "struct<map:int,array:string>",
    "array<int>=",
    "struct<a:int,b:array<int>>>",
    "array<array<int>>>>",
    "array<int>,",
    "array<int> int",
    "struct<a:int> x",
    "array<array<int>",
    "array<struct<a>>",
    "map<string,array<int>",
    "array<int>>>",
    "struct<a:int>,",
    "map<>",
    "array<array<>>",
    "struct<a:array<>>",
    "array<int> /* c */",
    "struct<a int Not Null Comment 'x', b string>",
    "array<array<foo>>",
    "array<struct<é:int>>",
    "struct<a:struct<b-c:int>>",
    "array<int>>,",
    "array<decimal(39,0)",
    "struct<a:int comment 'x'",
    "struct<a:int comment 'unterminated>",
    "array<string collate utf8_lcase>",
    "array<interval year>",
    "array<char(3)>",
    "struct<a:variant>",
    "map<string,int>/* c */",
    "struct<a:int comment r'raw\\n'>",
    "struct<a:int comment 'x' 'y'>",
    "struct<a:int comment 'line\\nbreak'>",
    "struct<a:int comment '\\u0041\\101'>",
]

DECIMALS = [(38, 38), (38, 0), (1, 0), (18, 18), (39, 0), (0, 0), (5, 6), (10, -1), (100, 0)]

LOCAL_PATH = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")


def local_spark_shell(spark_home: Path, script: str, data: str) -> list[str]:
    """Run the Scala script with a local `spark-shell -I`; returns its `@@` lines.

    The script reads its cases from the fixed path the container run uses.
    """
    files = {Path("/tmp/krishiv_script.scala"): script, Path("/tmp/krishiv_cases.jsonl"): data}
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    try:
        cmd = [
            str(spark_home / "bin" / "spark-shell"),
            "--master",
            "local[1]",
            "--driver-memory",
            "2g",
            "-I",
            "/tmp/krishiv_script.scala",
        ]
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, check=True, stdin=subprocess.DEVNULL
        ).stdout
    finally:
        for path in files:
            path.unlink(missing_ok=True)
    lines = [line for line in out.splitlines() if line.startswith("@@")]
    if "@@END" not in lines:
        raise SystemExit("spark-shell script did not complete:\n" + out[-2000:])
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="krishiv-bench-spark")
    ap.add_argument("--out", default="tests/fixtures/spark-4.2.0/type-strings.json")
    ap.add_argument(
        "--spark-home",
        help="run a local Spark distribution's spark-shell (for example a PySpark 4.2.0 install) "
        "instead of the container",
    )
    args = ap.parse_args()

    data = "".join(json.dumps({"ddl": d}) + "\n" for d in DDL)
    data += "".join(json.dumps({"precision": p, "scale": s}) + "\n" for p, s in DECIMALS)
    script = (Path(__file__).parent / "typestr" / "dump_types.scala").read_text(encoding="utf-8")
    if args.spark_home:
        lines = local_spark_shell(Path(args.spark_home), script, data)
        image = "local spark-shell"
    else:
        lines = spark_shell(args.container, script, data)
        image = container_image(args.container)
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
        "image": image,
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
