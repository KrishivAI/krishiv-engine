# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.xpath_float("a"), F.col("b"))
df = spark.sql("SELECT cosh(a) FROM t16")
df = df.select(F.xpath_short("a"), F.col("b"))
df = spark.sql("SELECT stack(a) FROM t16")
df = df.select(F.try_aes_decrypt("a"), F.col("b"))
