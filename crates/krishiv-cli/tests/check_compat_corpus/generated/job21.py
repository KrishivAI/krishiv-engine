# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT find_in_set(a) FROM t21")
df = df.select(F.variant_explode_outer("a"), F.col("b"))
df = df.select(F.legacy_fn_34("a"), F.col("b"))
df = df.select(F.legacy_fn_10("a"), F.col("b"))
df = df.select(F.randn("a"), F.col("b"))
df = df.select(F.log2("a"), F.col("b"))
