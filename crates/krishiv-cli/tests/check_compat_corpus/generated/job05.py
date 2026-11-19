# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.ilike("a"), F.col("b"))
df = spark.sql("SELECT sentences(a) FROM t5")
df = df.select(F.legacy_fn_12("a"), F.col("b"))
