# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT str_to_map(a) FROM t13")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.unix_date("a"), F.col("b"))
df = df.select(F.collate("a"), F.col("b"))
df = df.select(F.tuple_intersection_integer("a"), F.col("b"))
df = spark.sql("SELECT make_timestamp_ltz(a) FROM t13")
