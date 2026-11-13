# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT any(a) FROM t26")
df = df.select(F.kll_sketch_merge_double("a"), F.col("b"))
df = df.select(F.degrees("a"), F.col("b"))
df = df.select(F.legacy_fn_4("a"), F.col("b"))
