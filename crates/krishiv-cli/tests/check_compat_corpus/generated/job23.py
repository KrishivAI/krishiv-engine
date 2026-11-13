# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.string_agg("a"), F.col("b"))
df = df.select(F.xpath_int("a"), F.col("b"))
df = df.select(F.string("a"), F.col("b"))
df = df.select(F.legacy_fn_24("a"), F.col("b"))
df = df.select(F.substring("a"), F.col("b"))
