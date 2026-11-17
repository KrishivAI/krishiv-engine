# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT std(a) FROM t13")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.typeof("a"), F.col("b"))
df = df.select(F.collate("a"), F.col("b"))
df = df.select(F.tuple_sketch_estimate_double("a"), F.col("b"))
df = spark.sql("SELECT make_interval(a) FROM t13")
