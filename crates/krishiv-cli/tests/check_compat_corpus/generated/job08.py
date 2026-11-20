# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_1("a"), F.col("b"))
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t8")
df = spark.sql("SELECT xpath(a) FROM t8")
df = df.select(F.kll_sketch_merge_double("a"), F.col("b"))
df = df.select(F.get_json_object("a"), F.col("b"))
