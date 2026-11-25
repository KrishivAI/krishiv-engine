# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_14("a"), F.col("b"))
df = df.select(F.tuple_union_agg_double("a"), F.col("b"))
df = df.select(F.slice("a"), F.col("b"))
df = df.select(F.xpath_long("a"), F.col("b"))
