# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT stddev_pop(a) FROM t28")
df = df.select(F.var_samp("a"), F.col("b"))
df = df.select(F.legacy_fn_33("a"), F.col("b"))
