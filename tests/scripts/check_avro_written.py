# SPDX-License-Identifier: Apache-2.0
"""Spark 4.2 (spark-avro) reads the Avro files Krishiv wrote in each codec.

mkdir $D && KRISHIV_AVRO_DUMP=$D cargo test -p krishiv-io dump_written_files_for_spark -- --ignored
env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' python check_avro_written.py $D
"""

import glob
import os
import sys

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.master("local[2]")
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.13:4.2.0")
    .config("spark.sql.catalogImplementation", "in-memory")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
bad = 0
for path in sorted(glob.glob(f"{sys.argv[1]}/*.avro")):
    rows = spark.read.format("avro").load(path).collect()
    ok = len(rows) == 500 and all(r.v == f"v{r.id}" for r in rows)
    print("OK " if ok else "BAD", os.path.basename(path), len(rows))
    bad += not ok
spark.stop()
sys.exit(1 if bad else 0)
