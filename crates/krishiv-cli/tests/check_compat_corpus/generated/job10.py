# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_2(a) FROM t10")
df = df.select(F.btrim("a"), F.col("b"))
df = df.select(F.schema_of_variant_agg("a"), F.col("b"))
df = df.select(F.mask("a"), F.col("b"))
df = spark.sql("SELECT tuple_sketch_agg_integer(a) FROM t10")
