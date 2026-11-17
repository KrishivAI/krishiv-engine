# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_9("a"), F.col("b"))
df = df.select(F.tuple_union_theta_integer("a"), F.col("b"))
df = df.select(F.legacy_fn_15("a"), F.col("b"))
df = spark.sql("SELECT right(a) FROM t4")
df = df.select(F.date_add("a"), F.col("b"))
