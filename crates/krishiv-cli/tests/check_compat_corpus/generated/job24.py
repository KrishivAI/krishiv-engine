# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_sketch_get_n_bigint(a) FROM t24")
df = df.select(F.make_dt_interval("a"), F.col("b"))
df = spark.sql("SELECT exp(a) FROM t24")
df = df.select(F.inline("a"), F.col("b"))
