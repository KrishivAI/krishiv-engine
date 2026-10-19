#!/usr/bin/env python3
"""Generate Spark differential fixtures for the CSV reader and writer (P2-T081a).

Runs a REAL Spark (never from memory) in a throw-away container and records:

- `tokens`: how one CSV record splits into values under the tokenizer options (`sep`, `quote`, `escape`,
  `charToEscapeQuoteEscaping`, white-space trimming, `nullValue`/`emptyValue`, `unescapedQuoteHandling`),
  through `from_csv` into eight string columns plus `_corrupt_record`, over seeded combinations of tricky
  atoms;
- `reads`: whole files read with `spark.read.csv`: header, comments, empty lines, line separators, BOM,
  `multiLine`, parse modes with and without column pruning, `enforceSchema`, schema inference, value
  parsing of every type, datetime formats; each column cast to STRING in a UTC session, or the error
  classes (outermost first);
- `writes`: DataFrames written with `df.write.csv` under writer options; the single part file's text.

Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[1]' /work/gen_csv_goldens.py \
        > tests/fixtures/spark-4.2.0/csv.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import base64
import json
import os
import random
import re
import shutil
import sys
import tempfile

from pyspark.sql import SparkSession

SEED = 20261020
LINES_PER_SET = 160
COLS = [f"c{i}" for i in range(8)]
SCHEMA = ", ".join(f"{c} string" for c in COLS) + ", _corrupt_record string"
ATOMS = [
    "a", "", " ", "  b ", '"', '""', '"q"', '"a,b"', '"a""b"', '"a\\"b"', '"un', "\\", "\\\\", '"\\\\"', "'",
    "'s'", 'x"y', '"x"y', '"a" ', ' "a"', '"a"  ', "#c", "é", "NA", '"NA"', "E", "\t", ";", "a;b", '"a;b"',
    '"a\nb"', "||", "#", '"#"', "'a,b'", "'it''s'", '"a\\b"', "\\\"", '" "', "x ",
]
TOKEN_SETS = [
    {},
    {"quote": "'"},
    {"escape": '"'},
    {"sep": ";"},
    {"sep": "||"},
    {"sep": "\\t"},
    {"ignoreLeadingWhiteSpace": "true", "ignoreTrailingWhiteSpace": "true"},
    {"ignoreTrailingWhiteSpace": "true"},
    {"nullValue": "NA", "emptyValue": "E"},
    {"unescapedQuoteHandling": "STOP_AT_CLOSING_QUOTE"},
    {"unescapedQuoteHandling": "BACK_TO_DELIMITER"},
    {"unescapedQuoteHandling": "SKIP_VALUE"},
    {"unescapedQuoteHandling": "RAISE_ERROR"},
    {"charToEscapeQuoteEscaping": "#"},
    {"escape": "#"},
    {"quote": ""},
    {"escape": ""},
]


def classes(e):
    """The error classes in an exception's text, outermost first, without repeats."""
    out = []
    for c in re.findall(r"\[([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\]", str(e)):
        if c not in out:
            out.append(c)
    return out or [type(e).__name__]


def literal(s):
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def tokens(spark, rng):
    cases = []
    for opts in TOKEN_SETS:
        sep = opts.get("sep", ",").replace("\\t", "\t")
        lines = []
        for _ in range(LINES_PER_SET):
            n = rng.randint(1, 6)
            parts = [rng.choice(ATOMS) for _ in range(n)]
            joiner = sep if rng.random() < 0.85 else rng.choice([",", ";", "\t", "||"])
            lines.append(joiner.join(parts))
        options = ", ".join(f"{literal(k)}, {literal(v)}" for k, v in opts.items())
        expr = f"from_csv(s, '{SCHEMA}', map({options}))" if opts else f"from_csv(s, '{SCHEMA}')"
        df = spark.createDataFrame([(s,) for s in lines], "s string")
        try:
            rows = [r[1] for r in df.selectExpr("s", f"{expr} r").collect()]
            results = [[r[c] for c in [*COLS, "_corrupt_record"]] for r in rows]
        except Exception:  # noqa: BLE001 - a row failed: evaluate one by one
            results = []
            for s in lines:
                one = spark.createDataFrame([(s,)], "s string")
                try:
                    r = one.selectExpr(f"{expr} r").collect()[0][0]
                    results.append([r[c] for c in [*COLS, "_corrupt_record"]])
                except Exception as e:  # noqa: BLE001
                    results.append({"error": classes(e)})
        cases.append({"options": opts, "lines": lines, "rows": results})
    return cases


# (name, file text, read options, schema DDL or None to infer, selected columns or None)
READS = [
    ("header", "i,s,d\n1,x,2020-01-01\n\n   \n2,\"y,z\",2020-01-02\n", {"header": "true"}, "i int, s string, d date", None),
    ("no header infer", "1,x\n2,y\n", {}, None, None),
    ("header infer names", ",a,a,B,b, c ,\"q\"\n1,2,3,4,5,6,7\n", {"header": "true"}, None, None),
    ("header infer types", "a,b,c,d,e,f,g,h,k,l,m\n1,1,1.5,true,2020-01-01,2020-01-01 10:00:00,1e3,"
     "123456789012345678901,x,,2020-01-01T10:00:00Z\n2,3000000000,2,false,2020-1-2,2020-01-01T10:00:00.123,"
     "NaN,1,y,,2020-01-01 10:00\n", {"header": "true", "inferSchema": "true"}, None, None),
    ("infer more", "a,b,c,d,e,f,g,h\n1,2020-01-01,1.0,0.1,1e400,true,9223372036854775808,-1\n,2020-01-02,2,0.25,1,"
     "TRUE,1,+5\n3,,3,1.5E10,x,False,1,007\n", {"header": "true", "inferSchema": "true"}, None, None),
    ("infer dates", "x,y\n2020-01-01,2020-01-01\n2020-01-02,2020-01-02 10:00:00\n",
     {"header": "true", "inferSchema": "true"}, None, None),
    ("infer prefersDate false", "x\n2020-01-01\n", {"header": "true", "inferSchema": "true", "prefersDate": "false"},
     None, None),
    ("infer with timestampFormat", "x,y\n2020-01-01 10:00:00,2020-01-01\n",
     {"header": "true", "inferSchema": "true", "timestampFormat": "yyyy-MM-dd HH:mm:ss"}, None, None),
    ("infer with dateFormat", "x,y\n01/02/2020,2020-01-01\n",
     {"header": "true", "inferSchema": "true", "dateFormat": "MM/dd/yyyy"}, None, None),
    ("infer decimals", "a,b,c\n0.1,1.50,12345678901234567890.5\n", {"header": "true", "inferSchema": "true",
     "prefersDecimal": "true"}, None, None),
    ("infer nulls and nan", "a,b,c\nNA,NaN,Inf\n,1,-Inf\n", {"header": "true", "inferSchema": "true",
     "nullValue": "NA"}, None, None),
    ("infer empty", "", {"inferSchema": "true"}, None, None),
    ("infer header only", "a,b\n", {"header": "true", "inferSchema": "true"}, None, None),
    ("infer ragged", "a,b\n1\n1,2,3\n", {"header": "true", "inferSchema": "true"}, None, None),
    ("comment", "#c\na;b\n1;x\n# x\n2;\"y;z\"\n", {"sep": ";", "comment": "#", "header": "true"}, None, None),
    ("crlf", "1,a\r\n2,b\r3,c\n\r\n4,d", {}, "a int, b string", None),
    ("lineSep", "1,a;2,b;;3,c", {"lineSep": ";"}, "a int, b string", None),
    ("bom", "﻿a,b\n1,2\n", {"header": "true"}, None, None),
    ("bom no header", "﻿1,2\n", {}, "a int, b int", None),
    ("whitespace only", " \t \n1, 2 \n", {}, "a string, b string", None),
    ("multiline", "a,b\n1,\"x\ny\"\n  \n\n2,\"un\nterminated", {"multiLine": "true", "header": "true"}, None, None),
    ("multiline comment", "#c\n1,\"#x\n#y\"\n#z\n", {"multiLine": "true", "comment": "#"}, "a int, b string", None),
    ("multiline crlf", "1,\"x\r\ny\"\r\n2,z\r\n", {"multiLine": "true"}, "a int, b string", None),
]
MODE_TEXT = 'i,s,d\n1,x,2020-01-01\n2,y\n3,z,2020-01-03,extra\nbad,w,2020-01-04\n\n   \n5,"q,uoted",2020-01-05\n'
for mode in ["PERMISSIVE", "DROPMALFORMED", "FAILFAST"]:
    for corrupt in [False, True]:
        schema = "i int, s string, d date" + (", _corrupt_record string" if corrupt else "")
        for select in [None, ["i"], ["s"], ["d", "i"], []]:
            if corrupt and select == []:
                continue
            sel = select if not corrupt or select is None else [*select, "_corrupt_record"]
            READS.append((f"mode {mode} corrupt={corrupt} select={select}", MODE_TEXT,
                          {"header": "true", "mode": mode}, schema, sel))
READS += [
    ("custom corrupt column", "1,2\nx,3\n", {"columnNameOfCorruptRecord": "bad"}, "a int, b int, bad string", None),
    ("no pruning", "1,2\n3\n", {"mode": "DROPMALFORMED"}, "a int, b int", ["a"]),
    ("enforceSchema false ok", "A,b\n1,2\n", {"header": "true", "enforceSchema": "false"}, "a int, b int", None),
    ("enforceSchema false order", "A,b\n1,2\n", {"header": "true", "enforceSchema": "false"}, "b int, a int", None),
    ("enforceSchema false count", "a\n1\n", {"header": "true", "enforceSchema": "false"}, "a int, b int", None),
    ("enforceSchema true", "x,y\n1,2\n", {"header": "true"}, "a int, b int", None),
    ("integers", "v\n 1\n1 \n+1\n1.0\n0x10\n2147483648\n-0\n127\n-129\n", {"header": "true"}, None, None),
    ("integer types", "127,32767,2147483647,9223372036854775807\n128,32768,2147483648,9223372036854775808\n",
     {}, "a tinyint, b smallint, c int, d bigint", None),
    ("floats", "v\n1.5f\n 2.5 \n1e3\nInf\n-Inf\nNaN\nInfinity\n-Infinity\n1d\n.5\n5.\n0x1p3\n1e999\nnan\ninf\n",
     {"header": "true"}, "v double, w float", None),
    ("nan options", "nan,inf,-inf\nNaN,Inf,-Inf\n", {"nanValue": "nan", "positiveInf": "inf", "negativeInf": "-inf"},
     "a double, b double, c float", None),
    ("decimals", "v\n1.25\n-1.25\n12345.6\n1e1\n1.5f\n 2.5\n\"1,000.5\"\n.5\n5.\n", {"header": "true"}, "v decimal(6,1)",
     None),
    ("booleans", "v\ntrue\nTRUE\nTrue\nt\n1\n true\n", {"header": "true"}, "v boolean", None),
    ("dates", "v\n2020-01-01\n2020-1-1\n2020-01-01 10:00\n2020-01-01T10:00:00Z\n 2020-01-01\n2020\n2020-01\n20200101\n"
     "+2020-01-01\n2020-02-30\n2020-01-01x\n2020-01-01GMT\n", {"header": "true"}, "v date", None),
    ("date format", "v\n01/02/2020\n2020-01-02\n1/2/2020\n", {"header": "true", "dateFormat": "MM/dd/yyyy"}, "v date",
     None),
    ("date format fallback", "v\n01/02/2020\n2020-01-02\n", {"header": "true", "dateFormat": "MM/dd/yyyy",
     "enableDateTimeParsingFallback": "true"}, "v date", None),
    ("timestamps", "v\n2020-01-01\n2020-01-01 10:00:00\n2020-01-01T10:00:00.123456789\n2020-01-01 10:00:00+01:00\n"
     "2020-01-01T10:00:00Z\n 2020-01-01 10:00:00\n2020-01-01 10:00\n2020-01-01 10\n2020-01-01T\n2020-01-01 10:00:00UTC\n"
     "2020-01-01 10:00:00 +0100\n1577836800\n2020-01-01 10:00:00GMT\n2020-01-01 10:00:00 GMT+01:00\n",
     {"header": "true"}, "v timestamp", None),
    ("timestamps in a zone", "v\n2020-01-01 10:00:00\n2020-01-01T10:00:00Z\n", {"header": "true", "timeZone": "+05:30"},
     "v timestamp", None),
    ("timestamp format", "v\n2020-01-01 10:00:00\n2020-01-01\n02.01.2020 10:00\n",
     {"header": "true", "timestampFormat": "dd.MM.yyyy HH:mm"}, "v timestamp", None),
    ("ntz", "v\n2020-01-01\n2020-01-01 10:00:00.5\n2020-01-01 10:00:00+01:00\n10:00:00\n", {"header": "true"},
     "v timestamp_ntz", None),
    ("ntz format", "v\n02.01.2020 10:00\n02.01.2020 10:00 +01:00\n",
     {"header": "true", "timestampNTZFormat": "dd.MM.yyyy HH:mm[ XXX]"}, "v timestamp_ntz", None),
    ("binary", "v\nab\n[01 02]\n", {"header": "true"}, "v binary", None),
    ("nullValue", '1,NA,"NA",\n2,,"",x\n', {"nullValue": "NA"}, "a int, b string, c string, d string", None),
    ("emptyValue", '1,NA,"NA",\n2,,"",x\n', {"emptyValue": "E"}, "a int, b string, c string, d string", None),
    ("defaults", '1,NA,"NA",\n2,,"",x\n', {}, "a int, b string, c string, d string", None),
    ("maxColumns", "1,2,3\n", {"maxColumns": "2"}, "a string", None),
    ("maxCharsPerColumn", "12345,1\n", {"maxCharsPerColumn": "3"}, "a string, b string", None),
    ("latin1", "a,é\n", {"encoding": "ISO-8859-1"}, "a string, b string", None),
    ("count only", "1\n\n2\n#c\n", {"comment": "#"}, "a int", []),
    ("invalid utf8", None, {}, "a string, b string", None),
]


def read_case(spark, root, case):
    name, text, opts, schema, select = case
    path = os.path.join(root, re.sub(r"[^a-z0-9]+", "_", name.lower()))
    os.makedirs(path)
    if text is None:
        data = b"a,\xff\xc3\n"
    elif opts.get("encoding") == "ISO-8859-1":
        data = text.encode("latin-1")
    else:
        data = text.encode("utf-8")
    with open(os.path.join(path, "part.csv"), "wb") as f:
        f.write(data)
    out = {"name": name, "file": base64.b64encode(data).decode(), "options": opts, "schema": schema,
           "select": select}
    try:
        reader = spark.read.options(**opts)
        if schema is not None:
            reader = reader.schema(schema)
        df = reader.csv(path)
        out["inferred"] = df.schema.simpleString() if schema is None else None
        if select is not None:
            df = df.select(*select)
        if select == []:
            out["count"] = df.count()
        else:
            cols = [f"cast(`{c}` as string) as `{c}`" for c in df.columns]
            out["rows"] = [list(r) for r in df.selectExpr(*cols).collect()]
    except Exception as e:  # noqa: BLE001 - the error classes are the result
        out["error"] = classes(e)
    return out


# (name, SQL of the rows, writer options)
STRINGS = (
    "select * from values (1, 'plain'), (2, 'a,b'), (3, 'q\"uote'), (4, 'line\\nbreak'), (5, ''), (6, null), "
    "(7, ' lead'), (8, 'trail '), (9, 'back\\\\slash'), (10, '#hash'), (11, 'a\\rb'), (12, 'back\\\\slash,comma'), "
    "(13, ' \\t'), (14, '\"lead'), (15, '\\\\\"'), (16, 'é'), (17, 'x;y'), (18, 'NA') as t(i, s)"
)
TYPES = (
    "select * from values (1, 1.0D, cast(1.5 as float), cast(12.30 as decimal(10,2)), date'2020-01-02', "
    "timestamp'2020-01-02 03:04:05.123456', timestamp_ntz'2020-01-02 03:04:05.123456', true, X'AB01FF', "
    "cast(1 as tinyint), cast(2 as smallint), 3L, cast('1E-10' as decimal(38,10))), "
    "(2, 1e7D, cast('NaN' as float), cast(-0.5 as decimal(10,2)), date'0001-01-01', timestamp'2020-01-02 03:04:05', "
    "timestamp_ntz'2020-01-02 03:04:05', false, X'', null, null, null, cast(0 as decimal(38,10))), "
    "(3, 1e-4D, cast('Infinity' as float), null, date'9999-12-31', timestamp'0001-01-01 00:00:00', "
    "timestamp_ntz'0001-01-01 00:00:00.1', null, null, null, null, null, null), "
    "(4, cast('-Infinity' as double), cast(-0.0 as float), null, null, null, null, null, null, null, null, null, "
    "cast(-0.0000001 as decimal(38,10))), (5, cast('NaN' as double), cast(1e10 as float), null, null, null, null, "
    "null, null, null, null, null, null), (6, -0.0D, cast(0.001 as float), null, null, null, null, null, null, null, "
    "null, null, null), (7, 123456789.125D, cast(3.4e38 as float), null, null, null, null, null, null, null, null, "
    "null, null) as t(i, d, f, dec, dt, ts, ntz, b, bin, tiny, small, big, wide)"
)
WRITES = [
    ("strings default", STRINGS, {}),
    ("strings header", STRINGS, {"header": "true"}),
    ("strings quoteAll", STRINGS, {"quoteAll": "true"}),
    ("strings escapeQuotes false", STRINGS, {"escapeQuotes": "false"}),
    ("strings no trimming", STRINGS, {"ignoreLeadingWhiteSpace": "false", "ignoreTrailingWhiteSpace": "false"}),
    ("strings quote", STRINGS, {"quote": "'"}),
    ("strings escape quote", STRINGS, {"escape": '"'}),
    ("strings sep", STRINGS, {"sep": ";"}),
    ("strings multi sep", STRINGS, {"sep": "||"}),
    ("strings null empty", STRINGS, {"nullValue": "NA", "emptyValue": "E"}),
    ("strings lineSep", STRINGS, {"lineSep": "\r\n"}),
    ("strings charToEscape", STRINGS, {"charToEscapeQuoteEscaping": "#"}),
    ("strings no quote", STRINGS, {"quote": ""}),
    ("types default", TYPES, {"header": "true"}),
    ("types formats", TYPES, {"timestampFormat": "yyyy/MM/dd HH:mm:ss.SSSSSS XXX",
                              "timestampNTZFormat": "dd.MM.yyyy HH:mm:ss.SSSSSS", "dateFormat": "MM/dd/yy"}),
    ("types zone", TYPES, {"timeZone": "-08:00"}),
    ("header names", "select 1 as `a,b`, 2 as `q\"`, 3 as ` sp `", {"header": "true"}),
    ("empty with header", "select 1 as a, 'x' as b where false", {"header": "true"}),
    ("empty", "select 1 as a where false", {}),
]


def write_case(spark, root, case):
    name, sql, opts = case
    path = os.path.join(root, "w_" + re.sub(r"[^a-z0-9]+", "_", name.lower()))
    out = {"name": name, "sql": sql, "options": opts}
    try:
        spark.sql(sql).coalesce(1).write.options(**opts).csv(path)
        parts = sorted(p for p in os.listdir(path) if p.startswith("part-"))
        out["files"] = [p.split("-c000", 1)[1] for p in parts]
        with open(os.path.join(path, parts[0]), "rb") as f:
            out["content"] = base64.b64encode(f.read()).decode()
        out["schema"] = spark.sql(sql).schema.simpleString()
    except Exception as e:  # noqa: BLE001 - the error classes are the result
        out["error"] = classes(e)
    return out


def main() -> None:
    spark = SparkSession.builder.appName("csv-goldens").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    rng = random.Random(SEED)
    root = tempfile.mkdtemp()
    try:
        result = {
            "spark_version": spark.version,
            "seed": SEED,
            "tokens": tokens(spark, rng),
            "reads": [read_case(spark, root, c) for c in READS],
            "writes": [write_case(spark, root, c) for c in WRITES],
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)
    json.dump(result, sys.stdout, sort_keys=True, indent=0, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
