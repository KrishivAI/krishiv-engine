# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.log2("a"), F.col("b"))
df = spark.sql("SELECT octet_length(a) FROM t24")
df = df.select(F.legacy_fn_16("a"), F.col("b"))
df = df.select(F.day("a"), F.col("b"))
df = df.select(F.tuple_intersection_theta_integer("a"), F.col("b"))
