# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tuple_difference_double(a) FROM t16")
df = spark.sql("SELECT avg(a) FROM t16")
df = spark.sql("SELECT sentences(a) FROM t16")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.hypot("a"), F.col("b"))
