# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_json(a) FROM t28")
df = df.select(F.arrays_overlap("a"), F.col("b"))
df = df.select(F.tuple_intersection_integer("a"), F.col("b"))
