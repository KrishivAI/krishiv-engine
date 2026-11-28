# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT time_from_millis(a) FROM t8")
df = df.select(F.approx_top_k_estimate("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_merge_float(a) FROM t8")
df = df.select(F.shiftleft("a"), F.col("b"))
df = spark.sql("SELECT from_xml(a) FROM t8")
