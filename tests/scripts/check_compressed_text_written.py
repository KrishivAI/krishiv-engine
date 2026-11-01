# SPDX-License-Identifier: Apache-2.0
"""Spark 4.2 reads the CSV, JSON and text files Krishiv wrote in each codec.

Usage (local-mode pyspark, in a temp dir; see krishiv-internal CLAUDE.md):
  mkdir $D && KRISHIV_CODEC_DUMP=$D cargo test -p krishiv-io dump_written_files_for_spark -- --ignored
  env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' python check_compressed_text_written.py $D
Each file holds the same three rows, repeated three times (nine rows).
"""
import glob
import os
import sys

from pyspark.sql import SparkSession

d = sys.argv[1]
spark = (
    SparkSession.builder.master("local[2]")
    .config("spark.sql.catalogImplementation", "in-memory")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
bad = 0
for path in sorted(glob.glob(f"{d}/*")):
    fmt = os.path.basename(path).split("-")[0]
    if fmt == "csv":
        df = spark.read.schema("id int, s string, t string").option("multiLine", "true").csv(path)
    elif fmt == "json":
        df = spark.read.schema("id int, s string, t string").json(path)
    else:
        df = spark.read.text(path)
    rows = sorted(map(tuple, df.collect()), key=repr)
    ok = len(rows) == 12
    print("OK " if ok else "BAD", os.path.basename(path), len(rows), rows[:2])
    bad += not ok
spark.stop()
sys.exit(1 if bad else 0)
