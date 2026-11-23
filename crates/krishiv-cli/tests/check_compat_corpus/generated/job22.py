# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tuple_difference_integer(a) FROM t22")
df = spark.sql("SELECT bitmap_bucket_number(a) FROM t22")
df = df.select(F.legacy_fn_24("a"), F.col("b"))
df = df.select(F.st_asbinary("a"), F.col("b"))
