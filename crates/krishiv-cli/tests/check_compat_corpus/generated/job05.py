# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.inline("a"), F.col("b"))
df = spark.sql("SELECT reflect(a) FROM t5")
df = df.select(F.legacy_fn_12("a"), F.col("b"))
