# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_double("a"), F.col("b"))
df = df.select(F.percentile_approx("a"), F.col("b"))
df = spark.sql("SELECT tuple_union_theta_integer(a) FROM t19")
df = df.select(F.float("a"), F.col("b"))
df = df.select(F.tuple_sketch_agg_double("a"), F.col("b"))
