# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_36("a"), F.col("b"))
df = df.select(F.try_add("a"), F.col("b"))
df = spark.sql("SELECT theta_difference(a) FROM t17")
df = df.select(F.theta_sketch_estimate("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
