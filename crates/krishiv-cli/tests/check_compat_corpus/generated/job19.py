# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_intersection_agg_double("a"), F.col("b"))
df = df.select(F.float("a"), F.col("b"))
df = df.select(F.variant_get("a"), F.col("b"))
df = spark.sql("SELECT e(a) FROM t19")
