# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_intersection_agg_integer("a"), F.col("b"))
df = spark.sql("SELECT sin(a) FROM t20")
df = df.select(F.measure("a"), F.col("b"))
df = df.select(F.schema_of_json("a"), F.col("b"))
df = df.select(F.substring_index("a"), F.col("b"))
df = spark.sql("SELECT bround(a) FROM t20")
