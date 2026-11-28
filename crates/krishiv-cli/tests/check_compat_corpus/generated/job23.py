# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.approx_top_k_accumulate("a"), F.col("b"))
df = df.select(F.sentences("a"), F.col("b"))
df = spark.sql("SELECT tuple_union_integer(a) FROM t23")
df = spark.sql("SELECT legacy_fn_25(a) FROM t23")
df = df.select(F.tuple_sketch_summary_integer("a"), F.col("b"))
