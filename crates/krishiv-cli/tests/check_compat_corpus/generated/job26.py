# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_unix_timestamp(a) FROM t26")
df = df.select(F.make_timestamp("a"), F.col("b"))
df = spark.sql("SELECT hour(a) FROM t26")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t26")
