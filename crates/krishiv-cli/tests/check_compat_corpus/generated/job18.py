# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_36("a"), F.col("b"))
df = df.select(F.transform_values("a"), F.col("b"))
df = spark.sql("SELECT theta_sketch_agg(a) FROM t18")
df = df.select(F.try_reflect("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
