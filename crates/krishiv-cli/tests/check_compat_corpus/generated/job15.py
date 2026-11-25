# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT nullif(a) FROM t15")
df = df.select(F.schema_of_variant_agg("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
df = df.select(F.to_date("a"), F.col("b"))
df = df.select(F.std("a"), F.col("b"))
df = df.select(F.approx_top_k_estimate("a"), F.col("b"))
