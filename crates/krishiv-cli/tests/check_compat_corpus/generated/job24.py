# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.log1p("a"), F.col("b"))
df = spark.sql("SELECT nvl2(a) FROM t24")
df = df.select(F.legacy_fn_16("a"), F.col("b"))
df = df.select(F.day("a"), F.col("b"))
df = df.select(F.tuple_sketch_agg_integer("a"), F.col("b"))
