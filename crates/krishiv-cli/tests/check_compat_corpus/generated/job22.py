# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.try_validate_utf8("a"), F.col("b"))
df = df.select(F.tuple_difference_theta_integer("a"), F.col("b"))
df = spark.sql("SELECT approx_top_k_combine(a) FROM t22")
df = df.select(F.tuple_sketch_agg_integer("a"), F.col("b"))
