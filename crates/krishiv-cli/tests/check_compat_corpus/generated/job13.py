# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT stddev_pop(a) FROM t13")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.unhex("a"), F.col("b"))
df = df.select(F.collate("a"), F.col("b"))
df = df.select(F.tuple_intersection_theta_integer("a"), F.col("b"))
df = spark.sql("SELECT make_timestamp(a) FROM t13")
