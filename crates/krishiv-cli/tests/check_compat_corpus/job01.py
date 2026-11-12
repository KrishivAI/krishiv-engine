# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.time_to_micros("a"), F.col("b"))
df = spark.sql("SELECT lpad(a) FROM t1")
df = df.select(F.typeof("a"), F.col("b"))
df = spark.sql("SELECT filter(a) FROM t1")
