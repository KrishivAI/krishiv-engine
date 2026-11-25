# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_34(a) FROM t9")
df = spark.sql("SELECT to_time(a) FROM t9")
df = df.select(F.tuple_intersection_double("a"), F.col("b"))
df = spark.sql("SELECT vector_normalize(a) FROM t9")
df = df.select(F.typeof("a"), F.col("b"))
