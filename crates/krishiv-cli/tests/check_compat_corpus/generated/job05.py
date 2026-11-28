# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT approx_top_k(a) FROM t5")
df = spark.sql("SELECT xpath_float(a) FROM t5")
df = df.select(F.count_min_sketch("a"), F.col("b"))
df = df.select(F.regexp_extract("a"), F.col("b"))
df = df.select(F.st_asbinary("a"), F.col("b"))
df = df.select(F.reflect("a"), F.col("b"))
