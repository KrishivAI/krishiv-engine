# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_sketch_to_string_bigint(a) FROM t26")
df = df.select(F.regr_sxx("a"), F.col("b"))
df = df.select(F.legacy_fn_18("a"), F.col("b"))
df = df.select(F.degrees("a"), F.col("b"))
df = df.select(F.legacy_fn_4("a"), F.col("b"))
