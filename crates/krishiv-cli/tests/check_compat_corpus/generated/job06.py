# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.sum("a"), F.col("b"))
df = df.select(F.cardinality("a"), F.col("b"))
df = df.select(F.legacy_fn_18("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_32(a) FROM t6")
df = spark.sql("SELECT bitmap_bit_position(a) FROM t6")
