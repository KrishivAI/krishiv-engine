import os
import sys

from pyspark.sql import SparkSession

out = sys.argv[1]
spark = (
    SparkSession.builder.master("local[2]")
    .config("spark.jars.packages", "org.apache.spark:spark-avro_2.13:4.2.0")
    .config("spark.sql.catalogImplementation", "in-memory")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
q = """select 1 id, 'a' s, cast(1.5 as double) d, cast(2.5 as float) f, true b, cast('x' as binary) bin,
 date'2020-01-02' dt, timestamp'2020-01-02 03:04:05.678901' ts, timestamp_ntz'2020-01-02 03:04:05.678901' ntz,
 cast(12.34 as decimal(10,2)) dec, array(1,2,3) arr, map('k','v') m,
 named_struct('x',1,'y','z') st, cast(null as int) n,
 cast(5 as tinyint) ti, cast(6 as smallint) si, 9000000000 bg
 union all select 2,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null"""
df = spark.sql(q)
for c in ["uncompressed", "snappy", "deflate", "bzip2", "xz", "zstandard"]:
    try:
        df.coalesce(1).write.mode("overwrite").option("compression", c).format("avro").save(f"{out}/{c}")
        print("OK", c, [f for f in os.listdir(f"{out}/{c}") if f.endswith(".avro")])
    except Exception as e:
        print("ERR", c, str(e).splitlines()[0][:200])
df.show(truncate=False)
spark.stop()
