# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tuple_sketch_agg_integer(a) FROM t17")
df = df.select(F.dense_rank("a"), F.col("b"))
df = spark.sql("SELECT find_in_set(a) FROM t17")
df = spark.sql("SELECT arrays_overlap(a) FROM t17")
df = spark.sql("SELECT to_timestamp(a) FROM t17")
