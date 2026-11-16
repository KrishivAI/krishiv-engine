# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.url_decode("a"), F.col("b"))
df = df.select(F.kll_sketch_agg_float("a"), F.col("b"))
df = df.select(F.array_agg("a"), F.col("b"))
df = spark.sql("SELECT tuple_difference_double(a) FROM t6")
df = spark.sql("SELECT tuple_sketch_summary_integer(a) FROM t6")
