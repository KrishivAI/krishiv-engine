# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.string_agg("a"), F.col("b"))
df = spark.sql("SELECT levenshtein(a) FROM t1")
df = df.select(F.try_make_timestamp_ltz("a"), F.col("b"))
df = spark.sql("SELECT exists(a) FROM t1")
