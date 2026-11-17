# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_intersection_theta_double("a"), F.col("b"))
df = spark.sql("SELECT sign(a) FROM t20")
df = df.select(F.reflect("a"), F.col("b"))
df = df.select(F.rtrim("a"), F.col("b"))
df = df.select(F.substr("a"), F.col("b"))
df = spark.sql("SELECT bround(a) FROM t20")
