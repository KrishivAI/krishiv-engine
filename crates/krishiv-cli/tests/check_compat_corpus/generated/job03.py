# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_21("a"), F.col("b"))
df = spark.sql("SELECT variant_explode_outer(a) FROM t3")
df = df.select(F.greatest("a"), F.col("b"))
df = df.select(F.legacy_fn_18("a"), F.col("b"))
df = spark.sql("SELECT version(a) FROM t3")
