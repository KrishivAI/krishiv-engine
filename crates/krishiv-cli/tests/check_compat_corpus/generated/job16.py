# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT reflect(a) FROM t16")
df = spark.sql("SELECT avg(a) FROM t16")
df = spark.sql("SELECT kll_sketch_get_rank_float(a) FROM t16")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.is_valid_utf8("a"), F.col("b"))
