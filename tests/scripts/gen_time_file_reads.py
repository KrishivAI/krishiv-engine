#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Spark 4.3.0 expectations for the TIME type in the CSV and JSON file readers and writers.

Runs a REAL Spark (never from memory) in a throw-away container. It writes CSV and JSON files of TIME text
under a temp root, then records for each case the schema Spark reports and the rows (every column cast to
STRING in a UTC session, rows sorted), or the error (condition, the chain of conditions of a task failure).
Covered: `spark.sql.timeType.enabled` off (the 4.3.0 default: reads, writes and from_csv/from_json refuse,
to_csv/to_json evaluate), and on: a table of TIME texts read through CSV and JSON next to
`try_cast(text AS TIME(p))`, the parse modes, `timeFormat` on read and write, `nullValue`/`emptyValue`,
schema inference (TIME is never inferred), writes of TIME(p) columns read back as text, from_csv/from_json and
to_csv/to_json. The root is written as `<root>`. Usage (from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.3.0 \
        /opt/spark/bin/spark-submit --master 'local[1]' --driver-memory 1g /work/gen_time_file_reads.py \
        > tests/fixtures/spark-4.3.0/time-file-reads.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import json
import logging
import os
import re
import shutil
import tempfile

from pyspark.errors import PySparkException
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT_TOKEN = "<root>"
GATE = "spark.sql.timeType.enabled"

# TIME texts: the default grammar's edges (Spark's `stringToTime`), read through CSV and JSON and cast.
TEXTS = [
    "12:34:56",
    "12:34:56.123456789",
    "1:2:3",
    "23:59:59.999999999",
    "24:00:00",
    "bad",
    "12:34",
    " 12:34:56",
    "12:34:56 ",
    "\t12:34:56",
    "12:34:56.1234567891",
    "T12:34:56",
    "t12:34:56",
    "2024-01-01 12:34:56",
    "01:02:03 PM",
    "01:02:03 am",
    "12:34:56.",
    "12:60:00",
    "12:34:60",
    "00:00",
    "0:0",
    "123:00",
    "12:34:56.12",
    "12:34:56Z",
    "12:34:56+01:00",
    "12:34:56.123 PM",
    "13:00:00 PM",
    "12:00 AM",
    "00:00:00",
    "\uff11\uff12:34:56",
    "12:34:56.000000000",
    "12.34.56",
    "12-34-56",
    "1234",
    "12:34:5",
    " ",
    "12:34:56.123456789123456789",
]


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def layout(root):
    lines = "".join(f'{i},"{t}"\n' for i, t in enumerate(TEXTS))
    write_text(f"{root}/texts/a.csv", "id,t\n" + lines)
    write_text(f"{root}/texts_json/a.json", "".join(json.dumps({"id": i, "t": t}) + "\n" for i, t in enumerate(TEXTS)))
    # Non-string JSON tokens for a TIME field, and a JSON null.
    write_text(
        f"{root}/jtokens/a.json",
        '{"id":0,"t":45296}\n{"id":1,"t":true}\n{"id":2,"t":null}\n{"id":3,"t":12.5}\n{"id":4}\n'
        '{"id":5,"t":{"h":1}}\n{"id":6,"t":["12:00"]}\n',
    )
    # Empty field, a value equal to a custom nullValue.
    write_text(f"{root}/nulls/a.csv", 'id,t\n0,\n1,bad\n2,12:00\n3,""\n')
    # Texts in other layouts, for `timeFormat`.
    write_text(
        f"{root}/fmt/a.csv",
        "id,t\n0,12:34:56\n1,12.34.56\n2,1:05 PM\n3,12:34:56.123\n4,12:34:56.123456789\n5,2024-01-01 12:34:56\n"
        "6,12:34\n7,1:2:3\n",
    )
    write_text(
        f"{root}/fmt_json/a.json",
        '{"id":0,"t":"12:34:56"}\n{"id":1,"t":"12.34.56"}\n{"id":2,"t":"1:05 PM"}\n{"id":3,"t":"12:34:56.123"}\n'
        '{"id":4,"t":"12:34:56.123456789"}\n{"id":5,"t":"2024-01-01 12:34:56"}\n{"id":6,"t":"12:34"}\n'
        '{"id":7,"t":"1:2:3"}\n',
    )


def schema_json(df):
    return [{"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in df.schema.fields]


def outcome(spark, fn, root, conf):
    for key, value in conf.items():
        spark.conf.set(key, value)
    try:
        return _outcome(fn, root)
    finally:
        for key in conf:
            spark.conf.unset(key)


def _outcome(fn, root):
    try:
        df = fn()
        result = {"schema": schema_json(df)}
        names = [f"c{i}" for i in range(len(df.columns))]
        cast = df.toDF(*names).select([F.col(n).cast("string").alias(n) for n in names])
        rows = [[None if v is None else str(v) for v in r] for r in cast.collect()]
        rows.sort(key=lambda r: [("", "") if v is None else ("1", v) for v in r])
        result["rows"] = [[v.replace(root, ROOT_TOKEN) if v else v for v in r] for r in rows]
        return result
    except PySparkException as e:
        params = e.getMessageParameters() or {}
        out = {
            "error_class": e.getCondition() or "UNCLASSIFIED",
            "message_parameters": {k: v.replace(root, ROOT_TOKEN) for k, v in sorted(params.items())},
        }
        chain = _chain(str(e))
        if len(chain) > 1:
            out["error_chain"] = chain
        return out
    except Exception as e:  # a Py4J error PySpark did not convert (task failures)
        text = str(e)
        chain = _chain(text)
        return {"error_class": chain[0] if chain else "UNCLASSIFIED", "error_chain": chain}


def _chain(text):
    seen = []
    for m in re.finditer(r"\[([A-Z_]+(?:\.[A-Z_]+)*)\]", text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def cases(spark, root):
    """Yield (name, description, thunk, conf); thunks read `root`-relative data."""
    p = lambda sub: f"{root}/{sub}"  # noqa: E731
    on = {GATE: "true"}
    out = []

    def add(name, desc, thunk, conf=None):
        out.append((name, desc, thunk, on if conf is None else conf))

    def read(fmt, path, schema, **opts):
        if fmt == "csv":
            opts.setdefault("header", "true")
        return lambda: spark.read.schema(schema).options(**opts).format(fmt).load(p(path))

    def write_back(fmt, sql, **opts):
        # Write the query's rows as `fmt` into a fresh directory and read the files back as text lines.
        target = p(f"out/{len(out)}")

        def thunk():
            spark.sql(sql).coalesce(1).write.options(**opts).format(fmt).save(target)
            return spark.read.text(target)

        return thunk

    # 1. The gate off (Spark 4.3.0's default).
    off = {GATE: "false"}
    for fmt, path in (("csv", "texts"), ("json", "texts_json")):
        add(f"gate off {fmt} read TIME", f"{fmt} id INT, t TIME", read(fmt, path, "id INT, t TIME"), off)
        add(f"gate off {fmt} read TIME(3)", f"{fmt} id INT, t TIME(3)", read(fmt, path, "id INT, t TIME(3)"), off)
        add(f"gate off {fmt} write", f"{fmt} write of a TIME column", write_back(fmt, "SELECT TIME'01:02:03' t"), off)
    add("gate off from_csv", "from_csv TIME", lambda: spark.sql("SELECT from_csv('12:34:56', 't TIME') r"), off)
    add(
        "gate off from_json",
        "from_json TIME",
        lambda: spark.sql("""SELECT from_json('{"t":"12:34:56"}', 't TIME') r"""),
        off,
    )
    add(
        "gate off to_csv",
        "to_csv of a TIME literal",
        lambda: spark.sql("SELECT to_csv(named_struct('t', TIME'12:34:56.123')) r"),
        off,
    )
    add(
        "gate off to_json",
        "to_json of a TIME literal",
        lambda: spark.sql("SELECT to_json(named_struct('t', TIME'12:34:56.123')) r"),
        off,
    )

    # 2. The texts through CSV, JSON and a cast (the gate on).
    add(
        "texts cast",
        "try_cast(text AS TIME(p)) for each text, p in 0, 3, 6, 9",
        lambda: (
            spark.read.option("header", "true")
            .schema("id INT, s STRING")
            .csv(p("texts"))
            .selectExpr("id", "s", *[f"try_cast(s AS TIME({q})) AS t{q}" for q in (0, 3, 6, 9)])
        ),
    )
    for fmt, path in (("csv", "texts"), ("json", "texts_json")):
        for ty in ("TIME", "TIME(0)", "TIME(3)", "TIME(9)"):
            add(
                f"texts {fmt} {ty} PERMISSIVE",
                f"{fmt} id INT, t {ty}, _corrupt_record STRING",
                read(fmt, path, f"id INT, t {ty}, _corrupt_record STRING"),
            )
        add(
            f"texts {fmt} TIME(3) no corrupt column",
            f"{fmt} id INT, t TIME(3)",
            read(fmt, path, "id INT, t TIME(3)"),
        )
        add(
            f"texts {fmt} TIME(3) DROPMALFORMED",
            f"{fmt} id INT, t TIME(3) DROPMALFORMED",
            read(fmt, path, "id INT, t TIME(3)", mode="DROPMALFORMED"),
        )
        add(
            f"texts {fmt} TIME FAILFAST",
            f"{fmt} id INT, t TIME FAILFAST (first bad text is row 4, 24:00:00)",
            read(fmt, path, "id INT, t TIME", mode="FAILFAST"),
        )
    add(
        "json tokens TIME",
        "json numbers, booleans, objects, arrays and null for a TIME field",
        read("json", "jtokens", "id INT, t TIME, _corrupt_record STRING"),
    )
    # A value read into TIME(p) keeps its digits past p.
    add(
        "csv TIME(3) value past its precision",
        "csv TIME(3) row 1 compared, hour/second, typeof",
        lambda: (
            read("csv", "texts", "id INT, t TIME(3)")()
            .where("id = 1")
            .selectExpr("t = TIME'12:34:56.123' AS eq3", "t = TIME'12:34:56.123456789' AS eq9", "hour(t)", "typeof(t)")
        ),
    )

    # 3. timeFormat on read.
    for fmt, path in (("csv", "fmt"), ("json", "fmt_json")):
        for pattern in (
            "HH:mm:ss",
            "HH.mm.ss",
            "h:mm a",
            "HH:mm:ss.SSS",
            "HH:mm:ss.SSSSSSSSS",
            "yyyy-MM-dd HH:mm:ss",
            "H:m:s",
        ):
            add(
                f"timeFormat {fmt} {pattern}",
                f"{fmt} id INT, t TIME(9) with timeFormat {pattern}",
                read(fmt, path, "id INT, t TIME(9), _corrupt_record STRING", timeFormat=pattern),
            )
        add(
            f"timeFormat {fmt} invalid pattern",
            f"{fmt} timeFormat 'qqq HH'",
            read(fmt, path, "id INT, t TIME, _corrupt_record STRING", timeFormat="qqq HH"),
        )
        add(
            f"timeFormat {fmt} FAILFAST",
            f"{fmt} timeFormat HH:mm:ss FAILFAST",
            read(fmt, path, "id INT, t TIME", timeFormat="HH:mm:ss", mode="FAILFAST"),
        )

    # 4. nullValue and emptyValue.
    add("csv nulls default", "csv empty field and an empty quoted field", read("csv", "nulls", "id INT, t TIME"))
    add("csv nullValue", "csv nullValue 'bad'", read("csv", "nulls", "id INT, t TIME", nullValue="bad"))
    add("csv emptyValue", "csv emptyValue '12:00'", read("csv", "nulls", "id INT, t TIME", emptyValue="12:00"))

    # 5. Inference never gives TIME.
    add(
        "csv inferSchema",
        "csv inferSchema over TIME texts",
        lambda: spark.read.option("header", "true").option("inferSchema", "true").csv(p("fmt")),
    )
    add(
        "csv inferSchema timeFormat",
        "csv inferSchema with timeFormat HH:mm:ss",
        lambda: (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("timeFormat", "HH:mm:ss")
            .csv(p("fmt"))
        ),
    )
    add("json inference", "json inference over TIME texts", lambda: spark.read.json(p("fmt_json")))

    # 6. Writes, read back as text.
    sql = (
        "SELECT 1 AS id, CAST(TIME'12:34:56.123456789' AS TIME(9)) AS t9, CAST(TIME'12:34:56.120' AS TIME(3)) AS t3, "
        "CAST(TIME'01:02:03' AS TIME(6)) AS t6, CAST(TIME'00:00:00' AS TIME(0)) AS t0, CAST(NULL AS TIME) AS tn"
    )
    for fmt in ("csv", "json"):
        add(f"write {fmt}", f"{fmt} write of TIME(p) columns", write_back(fmt, sql))
        add(
            f"write {fmt} timeFormat",
            f"{fmt} write with timeFormat HH.mm.ss",
            write_back(fmt, sql, timeFormat="HH.mm.ss"),
        )
        add(
            f"write {fmt} timeFormat fraction",
            f"{fmt} write with timeFormat HH:mm:ss.SSS",
            write_back(fmt, sql, timeFormat="HH:mm:ss.SSS"),
        )
    add(
        "write csv read back TIME(3)",
        "a TIME(3) column read from CSV with 9 digits, written again",
        lambda: (
            read("csv", "texts", "id INT, t TIME(3)")().where("id = 1").coalesce(1).write.csv(p("out/rt")),
            spark.read.text(p("out/rt")),
        )[1],
    )

    # 7. from_csv, from_json, to_csv, to_json with the gate on.
    add(
        "from_csv",
        "from_csv TIME(3), TIME, a bad text, timeFormat",
        lambda: spark.sql(
            "SELECT from_csv('12:34:56.123456789', 't TIME(3)') a, from_csv('x', 't TIME') b, "
            "from_csv('12.34', 't TIME', map('timeFormat', 'HH.mm')) c"
        ),
    )
    add(
        "from_json",
        "from_json TIME(3), TIME, a bad text, timeFormat",
        lambda: spark.sql(
            """SELECT from_json('{"t":"12:34:56.123456789"}', 't TIME(3)') a, from_json('{"t":"x"}', 't TIME') b, """
            """from_json('{"t":"12.34"}', 't TIME', map('timeFormat', 'HH.mm')) c"""
        ),
    )
    add(
        "to_csv to_json",
        "to_csv/to_json of TIME(p), with and without timeFormat",
        lambda: spark.sql(
            "SELECT to_csv(named_struct('a', CAST(TIME'12:34:56.120' AS TIME(3)), 'b', TIME'01:02:03')) c1, "
            "to_csv(named_struct('a', TIME'12:34:56.5'), map('timeFormat', 'HH.mm')) c2, "
            "to_json(named_struct('a', CAST(TIME'12:34:56.123456789' AS TIME(9)))) j1, "
            "to_json(named_struct('a', TIME'12:34:56.5'), map('timeFormat', 'HH.mm')) j2"
        ),
    )
    return out


def main():
    logging.getLogger("py4j").setLevel(logging.ERROR)
    spark = (
        SparkSession.builder.appName("gen_time_file_reads")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    root = tempfile.mkdtemp(prefix="krishiv-time-io-")
    try:
        layout(root)
        results = []
        for name, desc, thunk, conf in cases(spark, root):
            entry = {"name": name, "description": desc, "conf": conf}
            entry.update(outcome(spark, thunk, root, conf))
            results.append(entry)
        doc = {
            "spark_version": spark.version,
            "generator": "tests/scripts/gen_time_file_reads.py",
            "session_time_zone": "UTC",
            "texts": TEXTS,
            "cases": results,
        }
        print(json.dumps(doc, indent=1, ensure_ascii=False))
    finally:
        shutil.rmtree(root, ignore_errors=True)
        spark.stop()


if __name__ == "__main__":
    main()
