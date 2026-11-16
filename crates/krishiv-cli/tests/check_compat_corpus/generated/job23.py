# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.str_to_map("a"), F.col("b"))
df = df.select(F.xpath_string("a"), F.col("b"))
df = df.select(F.stddev_samp("a"), F.col("b"))
df = df.select(F.legacy_fn_24("a"), F.col("b"))
df = df.select(F.struct("a"), F.col("b"))
