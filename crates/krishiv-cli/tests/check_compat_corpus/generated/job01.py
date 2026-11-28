# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_time(a) FROM t1")
df = df.select(F.xpath_float("a"), F.col("b"))
df = spark.sql("SELECT log2(a) FROM t1")
df = df.select(F.extract("a"), F.col("b"))
df = df.select(F.acosh("a"), F.col("b"))
