# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.kll_sketch_agg_bigint("a"), F.col("b"))
df = df.select(F.legacy_fn_9("a"), F.col("b"))
df = spark.sql("SELECT reflect(a) FROM t5")
df = spark.sql("SELECT typeof(a) FROM t5")
df = df.select(F.sinh("a"), F.col("b"))
