# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.schema_of_variant_agg("a"), F.col("b"))
df = spark.sql("SELECT floor(a) FROM t0")
df = df.select(F.legacy_fn_0("a"), F.col("b"))
df = spark.sql("SELECT explode(a) FROM t0")
df = spark.sql("SELECT time_from_seconds(a) FROM t0")
