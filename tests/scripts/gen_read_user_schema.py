#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Spark expectations for file reads with a user schema (KN-1 full fix).

Runs a REAL Spark (never from memory) in a throw-away container. It lays out Parquet, JSON, CSV and text
files and `k=v` partitioned directories under a temp root, then records for each case the schema Spark
reports for `spark.read.schema(...).<format>(path)` (names, types, nullability) and the rows (every column
cast to STRING in a UTC session, rows sorted), or the error (condition and message parameters). Partition
cases, nullability, parse modes of JSON, Parquet type mismatches, the text format and the SQL form
`CREATE TEMPORARY VIEW v (cols) USING <fmt> OPTIONS (path ...)` are covered; CSV with a user schema
(header, enforceSchema, parse modes) is already in `csv.json`. The root is written as `<root>`. Usage
(from repo root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[1]' --driver-memory 1g /work/gen_read_user_schema.py \
        > tests/fixtures/spark-4.2.0/read-user-schema.json

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
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

ROOT_TOKEN = "<root>"


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def layout(spark, root):
    # Parquet: a INT, b STRING, c BIGINT (all non-null), and a second tree with mixed-case names.
    spark.sql("SELECT 1 AS a, 'x' AS b, 10L AS c UNION ALL SELECT 2, 'y', 20L").coalesce(1).write.parquet(f"{root}/pq")
    spark.sql("SELECT 1 AS a, 'x' AS B").coalesce(1).write.parquet(f"{root}/pqcase")
    spark.sql("SELECT 1 AS a, 'x' AS b").coalesce(1).write.parquet(f"{root}/pqnum")
    # Partitioned Parquet: files hold only `v`; a second tree has `k` inside the data files too.
    for k in ("1", "2", "abc"):
        spark.sql(f"SELECT 'v{k}' AS v").coalesce(1).write.parquet(f"{root}/pp/p/k={k}")
        spark.sql(f"SELECT 'v{k}' AS v, 'in{k}' AS k").coalesce(1).write.parquet(f"{root}/ppk/p/k={k}")
    for k in ("1", "2"):
        spark.sql(f"SELECT 'v{k}' AS v").coalesce(1).write.parquet(f"{root}/pp1/p/k={k}")
        write_text(f"{root}/pc1/p/k={k}/f.csv", f"v{k},{k}0\n")
    # Partitioned CSV (no header).
    for k in ("1", "2", "abc"):
        write_text(f"{root}/pc/p/k={k}/f.csv", f"v{k},{k}0\n")
    # JSON.
    write_text(f"{root}/js/a.json", '{"a":1,"b":"x"}\n{"a":2}\n{"b":"z","a":3,"extra":9}\n')
    write_text(f"{root}/jbad/a.json", '{"a":1,"b":"x"}\n{"a":"x","b":"y"}\n{"a":1.5,"b":"z"}\n{"a":4,"b":{"q":1}}\n')
    write_text(f"{root}/jbad1/a.json", '{"a":1,"b":"x"}\n{"a":"x","b":"y"}\n')
    write_text(f"{root}/jbad2/a.json", '{"a":1,"b":"x"}\n{"a":1.5,"b":"y"}\n')
    write_text(f"{root}/jbad3/a.json", '{"a":1,"b":"x"}\n{"a":4,"b":{"q":1}}\n')
    write_text(f"{root}/jnest/a.json", '{"a":1,"s":{"p":1}}\n{"a":2}\n{"a":3,"s":{"p":2,"q":"t"}}\n')
    write_text(f"{root}/jcase/a.json", '{"a":1,"b":"x"}\n{"a":2,"b":"y"}\n')
    write_text(f"{root}/jmalformed/a.json", '{"a":1,"b":"x"}\nnot json\n{"a":3,"b":"z"}\n')
    # CSV, no header, and text.
    write_text(f"{root}/csv/a.csv", "1,x\n2,y\n")
    write_text(f"{root}/txt/a.txt", "line1\nline2\n")
    # Review cases: a partition column also in the data files at other positions and cases, typed partition values.
    for k in ("1", "2"):
        spark.sql(f"SELECT 'in{k}' AS k, 'v{k}' AS v").coalesce(1).write.parquet(f"{root}/kfirst/p/k={k}")
        spark.sql(f"SELECT 'a{k}' AS a, 'in{k}' AS k, 'v{k}' AS v").coalesce(1).write.parquet(f"{root}/kmid/p/k={k}")
        spark.sql(f"SELECT 'in{k}' AS K, 'v{k}' AS v").coalesce(1).write.parquet(f"{root}/kupper/p/k={k}")
        write_text(f"{root}/jkfirst/p/k={k}/a.json", f'{{"k":"in{k}","v":"v{k}"}}\n')
        write_text(f"{root}/tp/p/k={k}/a.txt", f"x{k}\n")
    # One value per tree where a case expects INVALID_PARTITION_VALUE: with several bad values, which one Spark
    # names follows its listing order, which varies between runs.
    for tree, k in (("pyear", "2024"), ("pabc", "abc"), ("pdec2", "1.50"), ("pdec2", "-3.25"), ("pone", "1")):
        spark.sql("SELECT 'v' AS v").coalesce(1).write.parquet(f"{root}/{tree}/p/k={k}")
    for tree, k in (("pfrac", "1.5"), ("pspace", " 1"), ("pplus", "+1"), ("pzeros", "007")):
        spark.sql("SELECT 'v' AS v").coalesce(1).write.parquet(f"{root}/{tree}/p/k={k}")
    for name, values in (
        ("pbool", ("true", "FALSE")),
        ("pdate", ("2024-01-02",)),
        ("pdef", ("__HIVE_DEFAULT_PARTITION__", "1")),
        ("pdec", ("1", "1.5", "-12")),
        ("pdbl", ("1.5", "-2", "1e3")),
    ):
        for k in values:
            spark.sql(f"SELECT 'v{k}' AS v").coalesce(1).write.parquet(f"{root}/{name}/p/k={k}")
            write_text(f"{root}/{name}c/p/k={k}/f.csv", f"v{k}\n")


def schema_json(df):
    return [{"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in df.schema.fields]


def outcome(fn, root, spark=None, conf=None):
    for key, value in (conf or {}).items():
        spark.conf.set(key, value)
    try:
        return _outcome(fn, root)
    finally:
        for key in conf or {}:
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
        return {
            "error_class": e.getCondition() or "UNCLASSIFIED",
            "message_parameters": {k: v.replace(root, ROOT_TOKEN) for k, v in sorted(params.items())},
        }
    except Exception as e:  # a Py4J error PySpark did not convert (task failures)
        text = str(e)
        chain = [m.group(1) for m in re.finditer(r"\[([A-Z_]+(?:\.[A-Z_]+)*)\]", text)]
        seen = []
        for c in chain:
            if c not in seen:
                seen.append(c)
        first = next((ln for ln in text.splitlines() if "[" in ln and "SparkException" in ln), "")
        return {
            "error_class": seen[0] if seen else "UNCLASSIFIED",
            "error_chain": seen,
            "message": re.sub(r"part-\d+-[0-9a-f-]+", "part-<uuid>", first.replace(root, ROOT_TOKEN))
            .lstrip(": ")
            .strip(),
        }


def nn(*fields):
    """A StructType from (name, type, nullable) triples."""
    return StructType([StructField(n, t, nl) for n, t, nl in fields])


def cases(spark, root):
    """Yield (name, description, thunk); thunks read `root`-relative data."""
    p = lambda sub: f"{root}/{sub}"  # noqa: E731
    out = []

    def add(name, desc, thunk, conf=None):
        out.append((name, desc, thunk, conf))

    def read(fmt, path, schema, **opts):
        return lambda: spark.read.schema(schema).options(**opts).format(fmt).load(p(path))

    # 1. Reported nullability.
    st_nn = nn(("a", IntegerType(), False), ("b", StringType(), False))
    ddl_nn = "a INT NOT NULL, b STRING NOT NULL"
    for fmt, path, extra in (("parquet", "pq", {}), ("json", "js", {}), ("csv", "csv", {})):
        add(
            f"nullability {fmt} StructType NOT NULL",
            f"{fmt} {path} schema a INT NOT NULL, b STRING NOT NULL",
            read(fmt, path, st_nn, **extra),
        )
        add(f"nullability {fmt} DDL NOT NULL", f"{fmt} {path} schema {ddl_nn}", read(fmt, path, ddl_nn, **extra))
    add(
        "nullability json NOT NULL with a null in data",
        "json js a INT NOT NULL, b STRING NOT NULL; record 2 has no b",
        read("json", "js", st_nn, mode="FAILFAST"),
    )
    add(
        "nullability parquet NOT NULL, FAILFAST irrelevant",
        "parquet pq a INT NOT NULL only",
        read("parquet", "pq", nn(("a", IntegerType(), False))),
    )

    # 2. JSON with a schema.
    ab = "a INT, b STRING"
    add("json missing and extra fields", f"json js {ab}", read("json", "js", ab))
    add("json wrong types PERMISSIVE", f"json jbad {ab}", read("json", "jbad", ab))
    add(
        "json wrong types PERMISSIVE with _corrupt_record",
        f"json jbad {ab}, _corrupt_record STRING",
        read("json", "jbad", ab + ", _corrupt_record STRING"),
    )
    add(
        "json wrong types DROPMALFORMED",
        f"json jbad {ab} DROPMALFORMED",
        read("json", "jbad", ab, mode="DROPMALFORMED"),
    )
    add("json wrong types FAILFAST", f"json jbad {ab} FAILFAST", read("json", "jbad", ab, mode="FAILFAST"))
    add("json string for int PERMISSIVE", f"json jbad1 {ab}", read("json", "jbad1", ab))
    add(
        "json string for int with _corrupt_record",
        f"json jbad1 {ab}, _corrupt_record STRING",
        read("json", "jbad1", ab + ", _corrupt_record STRING"),
    )
    add("json string for int FAILFAST", f"json jbad1 {ab} FAILFAST", read("json", "jbad1", ab, mode="FAILFAST"))
    add("json double for int PERMISSIVE", f"json jbad2 {ab}", read("json", "jbad2", ab))
    add(
        "json double for int with _corrupt_record",
        f"json jbad2 {ab}, _corrupt_record STRING",
        read("json", "jbad2", ab + ", _corrupt_record STRING"),
    )
    add("json double for int FAILFAST", f"json jbad2 {ab} FAILFAST", read("json", "jbad2", ab, mode="FAILFAST"))
    add("json object for string PERMISSIVE", f"json jbad3 {ab}", read("json", "jbad3", ab))
    add("json object for string FAILFAST", f"json jbad3 {ab} FAILFAST", read("json", "jbad3", ab, mode="FAILFAST"))
    add("json malformed line PERMISSIVE", f"json jmalformed {ab}", read("json", "jmalformed", ab))
    add(
        "json malformed line with _corrupt_record",
        f"json jmalformed {ab}, _corrupt_record STRING",
        read("json", "jmalformed", ab + ", _corrupt_record STRING"),
    )
    add(
        "json malformed line FAILFAST",
        f"json jmalformed {ab} FAILFAST",
        read("json", "jmalformed", ab, mode="FAILFAST"),
    )
    add(
        "json nested struct field absent from data",
        "json jnest a INT, s STRUCT<p: INT, q: STRING, r: INT>",
        read("json", "jnest", "a INT, s STRUCT<p: INT, q: STRING, r: INT>"),
    )
    add("json schema column absent from all data", "json js a INT, zz STRING", read("json", "js", "a INT, zz STRING"))
    add(
        "json name case differs, default caseSensitive",
        "json jcase A INT, B STRING",
        read("json", "jcase", "A INT, B STRING"),
    )
    add(
        "json name case differs, caseSensitive=true",
        "json jcase A INT, B STRING caseSensitive=true",
        read("json", "jcase", "A INT, B STRING"),
        {"spark.sql.caseSensitive": "true"},
    )

    # 3. Parquet with a schema.
    add("parquet subset", "parquet pq c BIGINT", read("parquet", "pq", "c BIGINT"))
    add("parquet reordered", "parquet pq c BIGINT, a INT", read("parquet", "pq", "c BIGINT, a INT"))
    add("parquet absent column", "parquet pq a INT, zz STRING", read("parquet", "pq", "a INT, zz STRING"))
    add(
        "parquet absent column NOT NULL",
        "parquet pq a INT, zz STRING NOT NULL",
        read("parquet", "pq", "a INT, zz STRING NOT NULL"),
    )
    add("parquet int read as string", "parquet pq a STRING", read("parquet", "pq", "a STRING"))
    add("parquet int read as bigint", "parquet pq a BIGINT", read("parquet", "pq", "a BIGINT"))
    add("parquet bigint read as int", "parquet pq c INT", read("parquet", "pq", "c INT"))
    add("parquet int read as decimal", "parquet pq a DECIMAL(10,2)", read("parquet", "pq", "a DECIMAL(10,2)"))
    add("parquet string read as int", "parquet pq b INT", read("parquet", "pq", "b INT"))
    add("parquet int read as double", "parquet pq a DOUBLE", read("parquet", "pq", "a DOUBLE"))
    add(
        "parquet name case differs, default",
        "parquet pqcase A INT, b STRING",
        read("parquet", "pqcase", "A INT, b STRING"),
    )
    add(
        "parquet name case differs, caseSensitive=true",
        "parquet pqcase A INT, b STRING caseSensitive=true",
        read("parquet", "pqcase", "A INT, b STRING"),
        {"spark.sql.caseSensitive": "true"},
    )

    # 4. Partitioned directories.
    for fmt, tree, base in (("parquet", "pp", "v STRING"), ("csv", "pc", "v STRING, w STRING")):
        names = base.split(", ")[0:1] if fmt == "parquet" else ["v", "w"]
        cols = [c.split(" ")[0] for c in base.split(", ")]
        del names
        add(f"partition {fmt} schema without k", f"{fmt} {tree}/p schema {base}", read(fmt, f"{tree}/p", base))
        add(
            f"partition {fmt} no schema",
            f"{fmt} {tree}/p, no user schema",
            lambda fmt=fmt, tree=tree: spark.read.format(fmt).load(p(f"{tree}/p")),
        )
        add(
            f"partition {fmt} k last",
            f"{fmt} {tree}/p schema {base}, k STRING",
            read(fmt, f"{tree}/p", base + ", k STRING"),
        )
        add(
            f"partition {fmt} k first",
            f"{fmt} {tree}/p schema k STRING, {base}",
            read(fmt, f"{tree}/p", "k STRING, " + base),
        )
        add(
            f"partition {fmt} k in the middle",
            f"{fmt} {tree}/p schema {cols[0]} STRING, k STRING" + (", w STRING" if fmt == "csv" else ""),
            read(fmt, f"{tree}/p", f"{cols[0]} STRING, k STRING" + (", w STRING" if fmt == "csv" else "")),
        )
        add(f"partition {fmt} k int", f"{fmt} {tree}/p schema {base}, k INT", read(fmt, f"{tree}/p", base + ", k INT"))
        add(
            f"partition {fmt} k int FAILFAST",
            f"{fmt} {tree}/p schema {base}, k INT FAILFAST",
            read(fmt, f"{tree}/p", base + ", k INT", mode="FAILFAST"),
        )
        add(
            f"partition {fmt} k bigint NOT NULL",
            f"{fmt} {tree}/p schema {base}, k BIGINT NOT NULL",
            read(fmt, f"{tree}/p", base + ", k BIGINT NOT NULL"),
        )
        add(f"partition {fmt} only k", f"{fmt} {tree}/p schema k STRING", read(fmt, f"{tree}/p", "k STRING"))
        add(
            f"partition {fmt} basePath",
            f"{fmt} {tree}/p/k=1 basePath={tree}/p schema {base}",
            lambda fmt=fmt, tree=tree, base=base: (
                spark.read.schema(base).option("basePath", p(f"{tree}/p")).format(fmt).load(p(f"{tree}/p/k=1"))
            ),
        )
        add(
            f"partition {fmt} basePath, k in schema",
            f"{fmt} {tree}/p/k=1 basePath={tree}/p schema {base}, k STRING",
            lambda fmt=fmt, tree=tree, base=base: (
                spark.read.schema(base + ", k STRING")
                .option("basePath", p(f"{tree}/p"))
                .format(fmt)
                .load(p(f"{tree}/p/k=1"))
            ),
        )
        add(
            f"partition {fmt} sub-directory without basePath",
            f"{fmt} {tree}/p/k=1 schema {base}",
            read(fmt, f"{tree}/p/k=1", base),
        )
    for fmt, tree, base in (("parquet", "pp1", "v STRING"), ("csv", "pc1", "v STRING, w STRING")):
        add(
            f"partition {fmt} values 1,2 no schema",
            f"{fmt} {tree}/p, no user schema",
            lambda fmt=fmt, tree=tree: spark.read.format(fmt).load(p(f"{tree}/p")),
        )
        add(
            f"partition {fmt} values 1,2 schema without k",
            f"{fmt} {tree}/p schema {base}",
            read(fmt, f"{tree}/p", base),
        )
        add(
            f"partition {fmt} values 1,2 k int last",
            f"{fmt} {tree}/p schema {base}, k INT",
            read(fmt, f"{tree}/p", base + ", k INT"),
        )
        add(
            f"partition {fmt} values 1,2 k int first",
            f"{fmt} {tree}/p schema k INT, {base}",
            read(fmt, f"{tree}/p", "k INT, " + base),
        )
        add(
            f"partition {fmt} values 1,2 k bigint NOT NULL",
            f"{fmt} {tree}/p schema {base}, k BIGINT NOT NULL",
            read(fmt, f"{tree}/p", base + ", k BIGINT NOT NULL"),
        )
        add(
            f"partition {fmt} values 1,2 k string NOT NULL",
            f"{fmt} {tree}/p schema {base}, k STRING NOT NULL",
            read(fmt, f"{tree}/p", base + ", k STRING NOT NULL"),
        )
        add(
            f"partition {fmt} values 1,2 k double",
            f"{fmt} {tree}/p schema {base}, k DOUBLE",
            read(fmt, f"{tree}/p", base + ", k DOUBLE"),
        )
    add(
        "partition parquet k also in data files, schema without k",
        "parquet ppk/p schema v STRING",
        read("parquet", "ppk/p", "v STRING"),
    )
    add(
        "partition parquet k also in data files, k STRING",
        "parquet ppk/p schema v STRING, k STRING",
        read("parquet", "ppk/p", "v STRING, k STRING"),
    )
    add(
        "partition parquet k also in data files, k INT",
        "parquet ppk/p schema v STRING, k INT",
        read("parquet", "ppk/p", "v STRING, k INT"),
    )
    add(
        "partition parquet k also in data files, no schema",
        "parquet ppk/p no schema",
        lambda: spark.read.parquet(p("ppk/p")),
    )

    # 5. text.
    add("text one column renamed", "text txt line STRING", read("text", "txt", "line STRING"))
    add("text default schema", "text txt no schema", lambda: spark.read.text(p("txt")))
    add("text two columns", "text txt a STRING, b STRING", read("text", "txt", "a STRING, b STRING"))
    add("text int column", "text txt line INT", read("text", "txt", "line INT"))
    add("text one column NOT NULL", "text txt line STRING NOT NULL", read("text", "txt", "line STRING NOT NULL"))

    # 6. SQL cross-check.
    def sql_view(name, cols, fmt, path, extra=""):
        def thunk():
            spark.sql(f"DROP VIEW IF EXISTS {name}")
            spark.sql(f"CREATE TEMPORARY VIEW {name} ({cols}) USING {fmt} OPTIONS (path '{p(path)}'{extra})")
            return spark.sql(f"SELECT * FROM {name}")

        return thunk

    add(
        "sql view json",
        f"CREATE TEMPORARY VIEW v1 ({ab}) USING json OPTIONS (path jbad)",
        sql_view("v1", ab, "json", "jbad"),
    )
    add(
        "sql view json NOT NULL",
        f"CREATE TEMPORARY VIEW v1b ({ddl_nn}) USING json OPTIONS (path js)",
        sql_view("v1b", ddl_nn, "json", "js"),
    )
    add(
        "sql view parquet",
        "CREATE TEMPORARY VIEW v2 (c BIGINT, zz STRING) USING parquet OPTIONS (path pq)",
        sql_view("v2", "c BIGINT, zz STRING", "parquet", "pq"),
    )
    add(
        "sql view parquet partitioned without k",
        "CREATE TEMPORARY VIEW v3 (v STRING) USING parquet OPTIONS (path pp/p)",
        sql_view("v3", "v STRING", "parquet", "pp/p"),
    )
    add(
        "sql view parquet partitioned with k",
        "CREATE TEMPORARY VIEW v4 (k STRING, v STRING) USING parquet OPTIONS (path pp/p)",
        sql_view("v4", "k STRING, v STRING", "parquet", "pp/p"),
    )
    add(
        "sql view csv partitioned with k int",
        "CREATE TEMPORARY VIEW v5 (v STRING, w STRING, k INT) USING csv OPTIONS (path pc/p)",
        sql_view("v5", "v STRING, w STRING, k INT", "csv", "pc/p"),
    )
    # 7. Review cases (KN-1 review): rules the cases above do not tell apart.
    for tree, label in (("kfirst", "k first"), ("kmid", "k in the middle"), ("kupper", "K upper-case")):
        add(
            f"overlap parquet {label}, no schema",
            f"parquet {tree}/p (data files hold k too), no user schema",
            lambda tree=tree: spark.read.parquet(p(f"{tree}/p")),
        )
    add(
        "overlap parquet K upper-case, no schema, caseSensitive=true",
        "parquet kupper/p no user schema caseSensitive=true",
        lambda: spark.read.parquet(p("kupper/p")),
        {"spark.sql.caseSensitive": "true"},
    )
    add(
        "overlap json k first, no schema",
        "json jkfirst/p (records hold k too), no user schema",
        lambda: spark.read.json(p("jkfirst/p")),
    )
    add(
        "overlap parquet k first, schema k,v",
        "parquet kfirst/p k STRING, v STRING",
        read("parquet", "kfirst/p", "k STRING, v STRING"),
    )
    add(
        "user K upper-case, caseSensitive=true",
        "parquet pp1/p K BIGINT, v STRING caseSensitive=true",
        read("parquet", "pp1/p", "K BIGINT, v STRING"),
        {"spark.sql.caseSensitive": "true"},
    )
    typed = [
        ("pfrac", "k=1.5", "INT"),
        ("pspace", "k= 1", "INT"),
        ("pplus", "k=+1", "INT"),
        ("pzeros", "k=007", "INT"),
        ("pfrac", "k=1.5", "DOUBLE"),
        ("pfrac", "k=1.5", "DECIMAL(5,2)"),
        ("pbool", "k=true,FALSE", "BOOLEAN"),
        ("pone", "k=1", "BOOLEAN"),
        ("pdate", "k=2024-01-02", "DATE"),
        ("pone", "k=1", "DATE"),
        ("pdef", "k=default,1", "INT"),
        ("pdec", "k=1,1.5,-12", "DECIMAL(5,2)"),
        ("pdec", "k=1,1.5,-12", "DECIMAL(10,0)"),
        ("pdbl", "k=1.5,-2,1e3", "DOUBLE"),
        ("pdbl", "k=1.5,-2,1e3", "FLOAT"),
        ("pp1", "k=1,2", "BINARY"),
        ("pp1", "k=1,2", "TINYINT"),
        ("pone", "k=1", "TIMESTAMP"),
    ]
    for tree, values, ty in typed:
        add(
            f"typed parquet {values} as {ty}",
            f"parquet {tree}/p schema v STRING, k {ty}",
            read("parquet", f"{tree}/p", f"v STRING, k {ty}"),
        )
    for tree, values, ty in (
        ("pdecc", "k=1,1.5,-12", "DECIMAL(5,2)"),
        ("pboolc", "k=true,FALSE", "BOOLEAN"),
        ("pdblc", "k=1.5,-2,1e3", "DOUBLE"),
    ):
        add(
            f"typed csv {values} as {ty}",
            f"csv {tree}/p schema v STRING, k {ty}",
            read("csv", f"{tree}/p", f"v STRING, k {ty}"),
        )
    add(
        "typed parquet k=abc as INT, validatePartitionColumns=false",
        "parquet pp/p schema v STRING, k INT validatePartitionColumns=false",
        read("parquet", "pp/p", "v STRING, k INT"),
        {"spark.sql.sources.validatePartitionColumns": "false"},
    )
    add(
        "typed parquet k=1.5 as INT, validatePartitionColumns=false",
        "parquet pfrac/p schema v STRING, k INT validatePartitionColumns=false",
        read("parquet", "pfrac/p", "v STRING, k INT"),
        {"spark.sql.sources.validatePartitionColumns": "false"},
    )
    # Second review pass: date shapes, and Parquet decimals whose written scale is or is not the column's.
    for tree, values, ty in (
        ("pyear", "k=2024", "DATE"),
        ("pyear", "k=2024", "TIMESTAMP"),
        ("pabc", "k=abc", "DATE"),
        ("pabc", "k=abc", "TIMESTAMP"),
        ("pdec2", "k=1.50,-3.25", "DECIMAL(5,2)"),
    ):
        add(
            f"typed parquet {values} as {ty}",
            f"parquet {tree}/p schema v STRING, k {ty}",
            read("parquet", f"{tree}/p", f"v STRING, k {ty}"),
        )
    add(
        "typed parquet k=1.5 as DECIMAL(5,2), where k = 1.5",
        "parquet pfrac/p schema v STRING, k DECIMAL(5,2) filter k = 1.5",
        lambda: read("parquet", "pfrac/p", "v STRING, k DECIMAL(5,2)")().where("k = 1.5"),
    )
    add(
        "typed parquet k=1.5 as DECIMAL(5,2), where k = 0.15",
        "parquet pfrac/p schema v STRING, k DECIMAL(5,2) filter k = 0.15",
        lambda: read("parquet", "pfrac/p", "v STRING, k DECIMAL(5,2)")().where("k = 0.15"),
    )
    add(
        "typed parquet k=1.50,-3.25 as DECIMAL(5,2), where k = 1.5",
        "parquet pdec2/p schema v STRING, k DECIMAL(5,2) filter k = 1.5",
        lambda: read("parquet", "pdec2/p", "v STRING, k DECIMAL(5,2)")().where("k = 1.5"),
    )
    add("text partitioned, schema line", "text tp/p schema line STRING", read("text", "tp/p", "line STRING"))
    add(
        "text partitioned, schema line,k",
        "text tp/p schema line STRING, k INT",
        read("text", "tp/p", "line STRING, k INT"),
    )
    add("duplicate column", "parquet pp1/p schema v STRING, v STRING", read("parquet", "pp1/p", "v STRING, v STRING"))
    add(
        "duplicate column, other case",
        "parquet pp1/p schema v STRING, V STRING",
        read("parquet", "pp1/p", "v STRING, V STRING"),
    )
    add(
        "duplicate column, other case, caseSensitive=true",
        "parquet pp1/p schema v STRING, V STRING caseSensitive=true",
        read("parquet", "pp1/p", "v STRING, V STRING"),
        {"spark.sql.caseSensitive": "true"},
    )
    add(
        "parquet name case differs, both upper, caseSensitive=true",
        "parquet pqcase A INT, B STRING caseSensitive=true",
        read("parquet", "pqcase", "A INT, B STRING"),
        {"spark.sql.caseSensitive": "true"},
    )
    return out


def main():
    spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    logging.getLogger("SQLQueryContextLogger").disabled = True
    root = tempfile.mkdtemp(prefix="rus")
    try:
        layout(spark, root)
        results = []
        for name, desc, thunk, conf in cases(spark, root):
            result = outcome(thunk, root, spark, conf)
            extra = {"conf": conf} if conf else {}
            results.append({"name": name, "input": desc.replace(root, ROOT_TOKEN), **extra, **result})
        doc = {
            "format_version": 1,
            "spark_version": spark.version,
            "generator": "tests/scripts/gen_read_user_schema.py",
            "notes": "Reads with a user schema on a tree laid out under <root>; reported schema plus rows "
            "(cast to STRING, UTC, sorted) or the error per case. Only Spark 4.2.0 was run: what 3.5, 4.0 and "
            "4.1 report for these reads (nullability, INVALID_PARTITION_VALUE, partition value conversion) is not "
            "pinned here.",
            "cases": results,
        }
        print(json.dumps(doc, indent=1, ensure_ascii=False))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
