# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT array_union(a) FROM t18")
df = df.select(F.acos("a"), F.col("b"))
df = spark.sql("SELECT tuple_intersection_theta_integer(a) FROM t18")
df = df.select(F.minute("a"), F.col("b"))
df = spark.sql("SELECT approx_top_k_accumulate(a) FROM t18")
