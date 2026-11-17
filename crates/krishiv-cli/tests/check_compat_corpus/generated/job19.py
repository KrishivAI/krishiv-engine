# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_sketch_theta_double("a"), F.col("b"))
df = spark.sql("SELECT session_user(a) FROM t19")
df = df.select(F.theta_sketch_estimate("a"), F.col("b"))
df = df.select(F.repeat("a"), F.col("b"))
df = df.select(F.std("a"), F.col("b"))
df = spark.sql("SELECT bitmap_construct_agg(a) FROM t19")
