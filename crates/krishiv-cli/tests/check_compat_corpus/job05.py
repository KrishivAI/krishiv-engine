# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_9("a"), F.col("b"))
df = df.select(F.tuple_sketch_agg_double("a"), F.col("b"))
df = df.select(F.legacy_fn_15("a"), F.col("b"))
df = spark.sql("SELECT signum(a) FROM t5")
df = df.select(F.dayname("a"), F.col("b"))
