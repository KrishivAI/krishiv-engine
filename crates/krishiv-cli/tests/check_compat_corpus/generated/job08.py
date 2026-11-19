# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_1("a"), F.col("b"))
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t8")
df = spark.sql("SELECT tuple_sketch_summary_double(a) FROM t8")
df = df.select(F.kll_sketch_get_n_bigint("a"), F.col("b"))
df = df.select(F.hex("a"), F.col("b"))
