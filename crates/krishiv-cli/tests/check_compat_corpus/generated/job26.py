# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT theta_intersection(a) FROM t26")
df = df.select(F.regexp_instr("a"), F.col("b"))
df = df.select(F.legacy_fn_18("a"), F.col("b"))
df = df.select(F.dayname("a"), F.col("b"))
df = df.select(F.legacy_fn_4("a"), F.col("b"))
