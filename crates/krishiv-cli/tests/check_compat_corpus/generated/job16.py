# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_sketch_agg_integer("a"), F.col("b"))
df = spark.sql("SELECT tuple_difference_theta_integer(a) FROM t16")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.hypot("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_22(a) FROM t16")
