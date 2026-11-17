# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_1("a"), F.col("b"))
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t7")
df = spark.sql("SELECT tuple_union_double(a) FROM t7")
df = df.select(F.kll_sketch_get_rank_float("a"), F.col("b"))
df = df.select(F.getbit("a"), F.col("b"))
