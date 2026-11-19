# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tuple_union_agg_integer(a) FROM t0")
df = df.select(F.legacy_fn_8("a"), F.col("b"))
df = spark.sql("SELECT schema_of_variant_agg(a) FROM t0")
