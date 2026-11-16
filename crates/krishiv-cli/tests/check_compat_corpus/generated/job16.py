# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT measure(a) FROM t16")
df = spark.sql("SELECT avg(a) FROM t16")
df = spark.sql("SELECT kll_sketch_get_rank_bigint(a) FROM t16")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.is_variant_null("a"), F.col("b"))
