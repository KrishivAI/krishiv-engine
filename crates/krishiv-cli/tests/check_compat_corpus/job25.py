# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.time_to_micros("a"), F.col("b"))
df = spark.sql("SELECT degrees(a) FROM t25")
df = df.select(F.csc("a"), F.col("b"))
df = spark.sql("SELECT weekofyear(a) FROM t25")
df = spark.sql("SELECT date_add(a) FROM t25")
df = df.select(F.nanvl("a"), F.col("b"))
