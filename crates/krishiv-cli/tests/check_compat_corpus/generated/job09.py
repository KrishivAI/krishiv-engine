# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_integer("a"), F.col("b"))
df = spark.sql("SELECT tuple_intersection_integer(a) FROM t9")
df = df.select(F.rand("a"), F.col("b"))
df = df.select(F.xpath_long("a"), F.col("b"))
df = df.select(F.legacy_fn_35("a"), F.col("b"))
