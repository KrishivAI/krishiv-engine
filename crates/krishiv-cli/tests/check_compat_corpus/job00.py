# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.try_aes_decrypt("a"), F.col("b"))
df = spark.sql("SELECT any_value(a) FROM t0")
df = df.select(F.substr("a"), F.col("b"))
df = df.select(F.xpath_float("a"), F.col("b"))
df = spark.sql("SELECT regr_slope(a) FROM t0")
