# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.kll_sketch_agg_double("a"), F.col("b"))
df = df.select(F.legacy_fn_9("a"), F.col("b"))
df = spark.sql("SELECT schema_of_avro(a) FROM t5")
df = spark.sql("SELECT try_variant_get(a) FROM t5")
df = df.select(F.sin("a"), F.col("b"))
