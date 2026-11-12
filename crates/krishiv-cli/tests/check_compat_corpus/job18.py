# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_36("a"), F.col("b"))
df = df.select(F.unix_micros("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_merge_double(a) FROM t18")
df = df.select(F.kll_sketch_to_string_float("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
