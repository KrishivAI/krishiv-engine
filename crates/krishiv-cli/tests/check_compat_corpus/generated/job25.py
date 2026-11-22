# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.log10("a"), F.col("b"))
df = spark.sql("SELECT nullifzero(a) FROM t25")
df = df.select(F.legacy_fn_16("a"), F.col("b"))
df = df.select(F.day("a"), F.col("b"))
df = df.select(F.tuple_sketch_summary_double("a"), F.col("b"))
