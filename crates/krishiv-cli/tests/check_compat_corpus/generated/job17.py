# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_36("a"), F.col("b"))
df = df.select(F.trim("a"), F.col("b"))
df = spark.sql("SELECT theta_intersection_agg(a) FROM t17")
df = df.select(F.theta_union_agg("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
