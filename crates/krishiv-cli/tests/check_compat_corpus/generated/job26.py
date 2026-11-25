# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_char(a) FROM t26")
df = df.select(F.make_time("a"), F.col("b"))
df = spark.sql("SELECT hll_union_agg(a) FROM t26")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t26")
