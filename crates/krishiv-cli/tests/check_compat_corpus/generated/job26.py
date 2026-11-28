# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.printf("a"), F.col("b"))
df = df.select(F.aggregate("a"), F.col("b"))
df = spark.sql("SELECT tuple_difference_integer(a) FROM t26")
