# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_36("a"), F.col("b"))
df = df.select(F.transform_keys("a"), F.col("b"))
df = spark.sql("SELECT theta_sketch_estimate(a) FROM t18")
df = df.select(F.tuple_difference_double("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
