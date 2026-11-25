# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_integer("a"), F.col("b"))
df = df.select(F.tuple_difference_double("a"), F.col("b"))
df = spark.sql("SELECT tuple_union_integer(a) FROM t15")
df = spark.sql("SELECT tuple_difference_integer(a) FROM t15")
