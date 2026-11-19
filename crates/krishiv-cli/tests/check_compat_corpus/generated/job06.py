# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.unix_micros("a"), F.col("b"))
df = df.select(F.kll_sketch_get_n_float("a"), F.col("b"))
df = df.select(F.array_agg("a"), F.col("b"))
df = spark.sql("SELECT tuple_intersection_agg_integer(a) FROM t6")
df = spark.sql("SELECT tuple_union_double(a) FROM t6")
