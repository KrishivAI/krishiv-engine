# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_varchar(a) FROM t26")
df = df.select(F.make_timestamp_ltz("a"), F.col("b"))
df = spark.sql("SELECT hypot(a) FROM t26")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t26")
