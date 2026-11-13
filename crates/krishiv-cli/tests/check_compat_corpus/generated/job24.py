# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_sketch_agg_float(a) FROM t24")
df = df.select(F.make_interval("a"), F.col("b"))
df = spark.sql("SELECT exp(a) FROM t24")
df = df.select(F.inline_outer("a"), F.col("b"))
