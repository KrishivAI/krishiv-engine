# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT getbit(a) FROM t7")
df = spark.sql("SELECT sinh(a) FROM t7")
df = df.select(F.kurtosis("a"), F.col("b"))
df = spark.sql("SELECT count_min_sketch(a) FROM t7")
df = df.select(F.legacy_fn_35("a"), F.col("b"))
