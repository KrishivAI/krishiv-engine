# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.ln("a"), F.col("b"))
df = df.select(F.tuple_sketch_estimate_double("a"), F.col("b"))
df = spark.sql("SELECT sin(a) FROM t20")
df = df.select(F.try_add("a"), F.col("b"))
df = df.select(F.schema_of_variant_agg("a"), F.col("b"))
df = df.select(F.min_by("a"), F.col("b"))
