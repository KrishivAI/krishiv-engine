# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_avro(a) FROM t1")
df = df.select(F.map_zip_with("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_merge_bigint(a) FROM t1")
