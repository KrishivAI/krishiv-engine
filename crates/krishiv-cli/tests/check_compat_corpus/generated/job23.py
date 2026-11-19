# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.shiftright("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_merge_float(a) FROM t23")
df = spark.sql("SELECT exp(a) FROM t23")
df = spark.sql("SELECT try_reflect(a) FROM t23")
