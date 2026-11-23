# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT element_at(a) FROM t4")
df = df.select(F.transform("a"), F.col("b"))
df = spark.sql("SELECT cot(a) FROM t4")
df = df.select(F.startswith("a"), F.col("b"))
df = df.select(F.tuple_difference_double("a"), F.col("b"))
