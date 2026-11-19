# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT stddev_samp(a) FROM t27")
df = df.select(F.variant_explode("a"), F.col("b"))
df = df.select(F.legacy_fn_33("a"), F.col("b"))
