# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT schema_of_avro(a) FROM t16")
df = spark.sql("SELECT avg(a) FROM t16")
df = spark.sql("SELECT kll_sketch_merge_bigint(a) FROM t16")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.int("a"), F.col("b"))
