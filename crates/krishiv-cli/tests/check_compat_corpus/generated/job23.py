# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT bitmap_bucket_number(a) FROM t23")
df = df.select(F.legacy_fn_24("a"), F.col("b"))
df = df.select(F.shiftleft("a"), F.col("b"))
