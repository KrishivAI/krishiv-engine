# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_intersection_theta_double("a"), F.col("b"))
df = spark.sql("SELECT theta_sketch_agg(a) FROM t9")
df = df.select(F.regexp_like("a"), F.col("b"))
df = df.select(F.tuple_sketch_theta_double("a"), F.col("b"))
df = df.select(F.legacy_fn_35("a"), F.col("b"))
