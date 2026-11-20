# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_23(a) FROM t0")
df = df.select(F.to_variant_object("a"), F.col("b"))
df = spark.sql("SELECT element_at(a) FROM t0")
df = df.select(F.curdate("a"), F.col("b"))
