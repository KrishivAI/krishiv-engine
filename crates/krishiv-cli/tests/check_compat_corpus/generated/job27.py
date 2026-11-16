# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT try_avg(a) FROM t27")
df = df.select(F.map_keys("a"), F.col("b"))
df = spark.sql("SELECT int(a) FROM t27")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t27")
