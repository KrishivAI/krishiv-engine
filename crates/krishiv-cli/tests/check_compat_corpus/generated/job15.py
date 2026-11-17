# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT theta_sketch_estimate(a) FROM t15")
df = spark.sql("SELECT avg(a) FROM t15")
df = spark.sql("SELECT reflect(a) FROM t15")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.ilike("a"), F.col("b"))
