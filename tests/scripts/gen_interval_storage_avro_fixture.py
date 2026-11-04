# SPDX-License-Identifier: Apache-2.0
"""Writes `tests/fixtures/spark-4.2.0/interval-storage-avro/spark.avro` with Spark 4.2.0.

Run in a fresh temporary directory:
    env -u SPARK_HOME uv run --no-project --python 3.12 --with 'pyspark==4.2.0' \\
        python tests/scripts/gen_interval_storage_avro_fixture.py <out-file>
"""

import glob
import shutil
import sys
import tempfile

from pyspark.sql import SparkSession

QUERY = """
select interval '1-2' year to month a, interval '3' month b, interval '1 02' day to hour c,
  interval '1 02:03:04.5' day to second d, interval '7' second e, array(interval '1' month) f,
  named_struct('x', interval '1' day) g, map('k', interval '2' hour) h, cast(null as interval day) n
"""

spark = (
    SparkSession.builder.master("local[1]")
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.13:4.2.0")
    .config("spark.sql.catalogImplementation", "in-memory")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
out = tempfile.mkdtemp()
spark.sql(QUERY).coalesce(1).write.mode("overwrite").format("avro").save(out + "/t")
shutil.copy(glob.glob(out + "/t/*.avro")[0], sys.argv[1])
