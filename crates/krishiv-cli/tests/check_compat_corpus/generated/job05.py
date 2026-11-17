# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.typeof("a"), F.col("b"))
df = df.select(F.kll_sketch_merge_double("a"), F.col("b"))
df = df.select(F.array_agg("a"), F.col("b"))
df = spark.sql("SELECT tuple_sketch_agg_integer(a) FROM t5")
df = spark.sql("SELECT xpath(a) FROM t5")
