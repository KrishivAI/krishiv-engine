# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT time_from_seconds(a) FROM t8")
df = df.select(F.approx_top_k_estimate("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_merge_double(a) FROM t8")
df = df.select(F.shiftright("a"), F.col("b"))
df = spark.sql("SELECT get(a) FROM t8")
