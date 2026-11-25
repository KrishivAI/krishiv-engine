# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_agg_double("a"), F.col("b"))
df = spark.sql("SELECT tuple_sketch_agg_integer(a) FROM t9")
df = df.select(F.overlay("a"), F.col("b"))
df = df.select(F.tuple_union_theta_double("a"), F.col("b"))
df = df.select(F.legacy_fn_35("a"), F.col("b"))
