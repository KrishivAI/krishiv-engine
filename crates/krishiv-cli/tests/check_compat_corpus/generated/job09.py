# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_double("a"), F.col("b"))
df = spark.sql("SELECT tuple_intersection_double(a) FROM t9")
df = df.select(F.randn("a"), F.col("b"))
df = df.select(F.xpath_int("a"), F.col("b"))
df = df.select(F.legacy_fn_35("a"), F.col("b"))
