# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_agg_double("a"), F.col("b"))
df = spark.sql("SELECT sequence(a) FROM t19")
df = df.select(F.theta_union_agg("a"), F.col("b"))
df = df.select(F.regr_syy("a"), F.col("b"))
df = df.select(F.stack("a"), F.col("b"))
df = spark.sql("SELECT bitmap_construct_agg(a) FROM t19")
