# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT trunc(a) FROM t27")
df = df.select(F.map_from_arrays("a"), F.col("b"))
df = spark.sql("SELECT instr(a) FROM t27")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t27")
