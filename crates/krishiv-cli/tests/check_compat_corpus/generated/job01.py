# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.hll_union_agg("a"), F.col("b"))
df = df.select(F.tuple_difference_double("a"), F.col("b"))
df = df.select(F.legacy_fn_23("a"), F.col("b"))
df = spark.sql("SELECT radians(a) FROM t1")
