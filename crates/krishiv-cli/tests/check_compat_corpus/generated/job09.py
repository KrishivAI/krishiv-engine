# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_34(a) FROM t9")
df = spark.sql("SELECT to_timestamp(a) FROM t9")
df = df.select(F.tuple_intersection_agg_integer("a"), F.col("b"))
df = spark.sql("SELECT vector_sum(a) FROM t9")
df = df.select(F.ucase("a"), F.col("b"))
