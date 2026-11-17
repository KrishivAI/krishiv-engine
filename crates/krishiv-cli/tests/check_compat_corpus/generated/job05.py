# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.initcap("a"), F.col("b"))
df = spark.sql("SELECT schema_of_variant_agg(a) FROM t5")
df = df.select(F.legacy_fn_12("a"), F.col("b"))
