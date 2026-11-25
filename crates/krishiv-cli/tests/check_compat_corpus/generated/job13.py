# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.try_reflect("a"), F.col("b"))
df = spark.sql("SELECT reflect(a) FROM t13")
df = df.select(F.asinh("a"), F.col("b"))
df = df.select(F.schema_of_variant("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_22(a) FROM t13")
