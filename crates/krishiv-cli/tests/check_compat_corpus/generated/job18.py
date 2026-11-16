# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_36("a"), F.col("b"))
df = df.select(F.try_parse_json("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_merge_float(a) FROM t18")
df = df.select(F.measure("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
