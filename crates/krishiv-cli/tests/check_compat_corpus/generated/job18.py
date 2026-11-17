# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_36("a"), F.col("b"))
df = df.select(F.try_make_timestamp_ntz("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_to_string_double(a) FROM t18")
df = df.select(F.reflect("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
