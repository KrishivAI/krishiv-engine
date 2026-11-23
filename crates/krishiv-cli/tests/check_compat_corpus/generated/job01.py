# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.schema_of_csv("a"), F.col("b"))
df = spark.sql("SELECT len(a) FROM t1")
df = df.select(F.parse_json("a"), F.col("b"))
df = df.select(F.approx_top_k_combine("a"), F.col("b"))
df = spark.sql("SELECT rlike(a) FROM t1")
