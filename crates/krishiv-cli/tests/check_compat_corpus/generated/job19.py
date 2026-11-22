# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_difference_theta_integer("a"), F.col("b"))
df = df.select(F.floor("a"), F.col("b"))
df = df.select(F.vector_avg("a"), F.col("b"))
df = spark.sql("SELECT element_at(a) FROM t19")
