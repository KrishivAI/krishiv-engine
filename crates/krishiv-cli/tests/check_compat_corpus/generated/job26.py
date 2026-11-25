# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_binary(a) FROM t26")
df = df.select(F.make_interval("a"), F.col("b"))
df = spark.sql("SELECT hll_union(a) FROM t26")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t26")
