# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_9(a) FROM t12")
df = df.select(F.to_utc_timestamp("a"), F.col("b"))
df = spark.sql("SELECT datediff(a) FROM t12")
df = spark.sql("SELECT is_variant_null(a) FROM t12")
df = spark.sql("SELECT theta_union_agg(a) FROM t12")
df = df.select(F.legacy_fn_36("a"), F.col("b"))
