#!/usr/bin/env python3
"""Spark 4.2 CSV schema inference with `samplingRatio` (P2-T081a).

A file of 300 rows and 100 integer columns where column `j` holds one string, in row `j`: the inferred type of
column `j` is `string` exactly when Spark sampled row `j`, so every column reveals one draw of the sampler. The
file is read with several ratios and partition layouts (Spark seeds each partition's sampler with its index).

Runs against a Spark 4.2 Connect server (2 cores), which writes and reads the file on its own file system:

    SPARK_HOME=<empty dir> uv run --no-project --python 3.12 --with 'pyspark-client==4.2.0' \
        python tests/scripts/gen_csv_sampling.py sc://<host>:15002 > tests/fixtures/spark-4.2.0/csv-sampling.json
"""

import json
import sys
import uuid

from pyspark.sql import SparkSession

ROWS = 300
COLS = 100
# (samplingRatio, maxPartitionBytes, openCostInBytes): one partition, then many small ones.
LAYOUTS = [
    (1.0, None, None),
    (0.5, None, None),
    (0.1, None, None),
    (0.9, None, None),
    (0.5, 4000, 0),
    (0.3, 4000, 0),
    (0.5, 9000, 0),
]


def main() -> None:
    spark = SparkSession.builder.remote(sys.argv[1]).getOrCreate()
    lines = [",".join("x" if r == c else "1" for c in range(COLS)) for r in range(ROWS)]
    path = f"/tmp/krishiv-csv-sampling-{uuid.uuid4().hex}"
    spark.createDataFrame([(line,) for line in lines], "value string").coalesce(1).write.text(path)
    cases = []
    for ratio, max_bytes, open_cost in LAYOUTS:
        spark.conf.set("spark.sql.files.maxPartitionBytes", str(max_bytes or 134217728))
        spark.conf.set("spark.sql.files.openCostInBytes", str(open_cost if open_cost is not None else 4194304))
        df = spark.read.option("inferSchema", "true").option("samplingRatio", str(ratio)).csv(path)
        cases.append(
            {
                "ratio": ratio,
                "maxPartitionBytes": max_bytes,
                "openCostInBytes": open_cost,
                "types": [f.dataType.simpleString() for f in df.schema.fields],
            }
        )
    json.dump(
        {"spark_version": spark.version, "rows": ROWS, "cols": COLS, "cases": cases},
        sys.stdout,
        sort_keys=True,
        indent=0,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
