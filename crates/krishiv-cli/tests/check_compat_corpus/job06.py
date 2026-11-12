# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.version("a"), F.col("b"))
df = df.select(F.kll_sketch_agg_double("a"), F.col("b"))
df = df.select(F.array_agg("a"), F.col("b"))
df = spark.sql("SELECT to_protobuf(a) FROM t6")
df = spark.sql("SELECT tuple_sketch_estimate_integer(a) FROM t6")
