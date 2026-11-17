# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tuple_union_theta_integer(a) FROM t3")
df = df.select(F.kll_merge_agg_float("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_9(a) FROM t3")
df = df.select(F.map_from_entries("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_2(a) FROM t3")
df = spark.sql("SELECT kll_sketch_to_string_bigint(a) FROM t3")
