# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_intersection_integer("a"), F.col("b"))
df = df.select(F.radians("a"), F.col("b"))
df = df.select(F.array_except("a"), F.col("b"))
