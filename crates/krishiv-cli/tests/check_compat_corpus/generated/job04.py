# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_9("a"), F.col("b"))
df = df.select(F.xpath_boolean("a"), F.col("b"))
df = df.select(F.legacy_fn_15("a"), F.col("b"))
df = spark.sql("SELECT reverse(a) FROM t4")
df = df.select(F.date_add("a"), F.col("b"))
