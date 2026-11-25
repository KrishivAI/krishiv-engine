# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.to_utc_timestamp("a"), F.col("b"))
df = df.select(F.try_reflect("a"), F.col("b"))
df = df.select(F.array_agg("a"), F.col("b"))
df = spark.sql("SELECT tuple_sketch_summary_integer(a) FROM t6")
df = spark.sql("SELECT tuple_union_theta_integer(a) FROM t6")
