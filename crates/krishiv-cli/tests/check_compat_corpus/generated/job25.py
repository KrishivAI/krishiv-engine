# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT any(a) FROM t25")
df = df.select(F.tuple_intersection_agg_double("a"), F.col("b"))
df = df.select(F.day("a"), F.col("b"))
df = df.select(F.legacy_fn_4("a"), F.col("b"))
