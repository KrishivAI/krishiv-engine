# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.schema_of_json("a"), F.col("b"))
df = spark.sql("SELECT len(a) FROM t1")
df = df.select(F.parse_url("a"), F.col("b"))
df = df.select(F.approx_top_k_combine("a"), F.col("b"))
df = spark.sql("SELECT round(a) FROM t1")
