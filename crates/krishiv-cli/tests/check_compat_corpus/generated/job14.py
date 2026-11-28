# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_24(a) FROM t14")
df = spark.sql("SELECT var_pop(a) FROM t14")
df = spark.sql("SELECT try_reflect(a) FROM t14")
df = spark.sql("SELECT legacy_fn_21(a) FROM t14")
df = spark.sql("SELECT schema_of_variant_agg(a) FROM t14")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t14")
