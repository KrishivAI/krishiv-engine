# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.try_validate_utf8("a"), F.col("b"))
df = df.select(F.kll_sketch_merge_float("a"), F.col("b"))
df = df.select(F.array_agg("a"), F.col("b"))
df = spark.sql("SELECT tuple_sketch_estimate_integer(a) FROM t5")
df = spark.sql("SELECT xpath_int(a) FROM t5")
