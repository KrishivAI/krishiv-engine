#!/usr/bin/env python3
"""Generate Spark expectations for reads by path (P2-T154).

Runs a REAL Spark (never from memory) in a throw-away container. It lays out a directory tree under a
root (Spark-written Parquet directories with their `_SUCCESS` and `.crc` files, an empty directory, a
directory holding only hidden files, hidden files next to a data file), then records, for each case, the
schema Spark reads (names, types, nullability) or the error it raises (condition and message
parameters). Cases are DataFrame reads (`spark.read.parquet(paths)`) and SQL (`` parquet.`path` ``).
The root is written as `<root>` so the Krishiv test can lay out the same tree anywhere. Usage (from repo
root):

    docker run --rm -v "$PWD/tests/scripts:/work:ro" apache/spark:4.2.0 \
        /opt/spark/bin/spark-submit --master 'local[1]' /work/gen_file_read_probe.py \
        > tests/fixtures/spark-4.2.0/file-read-probe.json

The fixture goes to stdout (the container user cannot write the mount); Spark logs go to stderr.
"""

import json
import logging
import os
import shutil
import tempfile

from pyspark.errors import PySparkException
from pyspark.sql import SparkSession

ROOT_TOKEN = "<root>"

# (name, kind, argument): "read" takes a list of paths, "sql" a statement; `<root>` is the tree's root.
CASES = [
    ("directory", "read", ["<root>/one"]),
    ("hidden files are skipped", "read", ["<root>/h"]),
    ("missing", "read", ["<root>/nope"]),
    ("one path missing", "read", ["<root>/one", "<root>/nope"]),
    ("empty directory", "read", ["<root>/empty"]),
    ("only hidden files", "read", ["<root>/onlyhidden"]),
    ("hidden file named", "read", ["<root>/h/.hidden.parquet"]),
    ("no paths", "read", []),
    ("file URL", "read", ["file://<root>/one"]),
    ("sql path", "sql", "SELECT * FROM parquet.`<root>/one`"),
    ("sql upper-case format", "sql", "SELECT * FROM PARQUET.`<root>/one`"),
    ("sql missing", "sql", "SELECT * FROM parquet.`<root>/nope`"),
    ("sql other file format, missing", "sql", "SELECT * FROM json.`<root>/nope`"),
    ("sql unknown format", "sql", "SELECT * FROM foo.`<root>/one`"),
    ("sql non-file source", "sql", "SELECT * FROM jdbc.`<root>/one`"),
    ("sql three parts", "sql", "SELECT * FROM spark_catalog.parquet.`<root>/one`"),
    ("sql path as qualifier", "sql", "SELECT `<root>/one`.a FROM parquet.`<root>/one`"),
    ("sql format as qualifier", "sql", "SELECT parquet.a FROM parquet.`<root>/one`"),
    ("sql alias as qualifier", "sql", "SELECT t.a FROM parquet.`<root>/one` t"),
    ("sql runSQLOnFiles off", "sql-off", "SELECT * FROM parquet.`<root>/one`"),
]


def layout(spark, root):
    spark.sql("SELECT 1 AS a, 'x' AS B, CAST(NULL AS INT) AS c").write.parquet(f"{root}/one")
    spark.sql("SELECT 1 AS a").write.parquet(f"{root}/h")
    with open(f"{root}/h/.hidden.parquet", "w") as f:
        f.write("junk")
    with open(f"{root}/h/_junk", "w") as f:
        f.write("junk")
    os.makedirs(f"{root}/empty")
    os.makedirs(f"{root}/onlyhidden")
    open(f"{root}/onlyhidden/_SUCCESS", "w").close()


def outcome(fn, root):
    try:
        df = fn()
        return {
            "schema": [
                {"name": f.name, "type": f.dataType.simpleString(), "nullable": f.nullable} for f in df.schema.fields
            ]
        }
    except PySparkException as e:
        params = e.getMessageParameters() or {}
        return {
            "error_class": e.getCondition() or "UNCLASSIFIED",
            "message_parameters": {k: v.replace(root, ROOT_TOKEN) for k, v in sorted(params.items())},
        }


def main():
    spark = SparkSession.builder.getOrCreate()
    # PySpark logs the errors this probe provokes to stdout, where the fixture goes.
    logging.getLogger("SQLQueryContextLogger").disabled = True
    root = tempfile.mkdtemp(prefix="probe")
    try:
        layout(spark, root)
        results = []

        def sub(text):
            return text.replace(ROOT_TOKEN, root)

        for name, kind, arg in CASES:
            if kind == "read":
                paths = [sub(p) for p in arg]
                result = outcome(lambda paths=paths: spark.read.parquet(*paths), root)
            else:
                spark.conf.set("spark.sql.runSQLOnFiles", "false" if kind == "sql-off" else "true")
                query = sub(arg)
                result = outcome(lambda query=query: spark.sql(query), root)
                spark.conf.unset("spark.sql.runSQLOnFiles")
            results.append({"name": name, "kind": kind, "input": arg, **result})
        doc = {
            "format_version": 1,
            "spark_version": spark.version,
            "generator": "tests/scripts/gen_file_read_probe.py",
            "notes": "Reads by path on a tree laid out under <root>; schema or error per case.",
            "cases": results,
        }
        print(json.dumps(doc, indent=1, ensure_ascii=False))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
