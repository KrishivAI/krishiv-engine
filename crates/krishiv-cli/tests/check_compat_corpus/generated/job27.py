# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.filter("a"), F.col("b"))
df = df.select(F.xpath_string("a"), F.col("b"))
df = spark.sql("SELECT date_format(a) FROM t27")
df = spark.sql("SELECT count_min_sketch(a) FROM t27")
