# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_agg_double("a"), F.col("b"))
df = spark.sql("SELECT tuple_intersection_agg_double(a) FROM t8")
df = df.select(F.random("a"), F.col("b"))
df = df.select(F.xpath_double("a"), F.col("b"))
df = df.select(F.legacy_fn_35("a"), F.col("b"))
