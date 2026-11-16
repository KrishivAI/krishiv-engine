# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_intersection_agg_integer("a"), F.col("b"))
df = spark.sql("SELECT theta_intersection(a) FROM t9")
df = df.select(F.regexp_substr("a"), F.col("b"))
df = df.select(F.tuple_sketch_estimate_integer("a"), F.col("b"))
df = df.select(F.legacy_fn_35("a"), F.col("b"))
