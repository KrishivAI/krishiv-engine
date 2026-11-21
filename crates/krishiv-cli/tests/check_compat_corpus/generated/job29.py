# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_25(a) FROM t29")
df = spark.sql("SELECT regexp_replace(a) FROM t29")
df = df.select(F.bit_get("a"), F.col("b"))
df = df.select(F.count_min_sketch("a"), F.col("b"))
