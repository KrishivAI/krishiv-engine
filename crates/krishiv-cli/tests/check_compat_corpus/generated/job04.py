# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT e(a) FROM t4")
df = df.select(F.to_xml("a"), F.col("b"))
df = spark.sql("SELECT cot(a) FROM t4")
df = df.select(F.stack("a"), F.col("b"))
df = df.select(F.tuple_difference_integer("a"), F.col("b"))
