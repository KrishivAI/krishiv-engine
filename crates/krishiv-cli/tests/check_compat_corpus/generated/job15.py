# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT theta_union_agg(a) FROM t15")
df = spark.sql("SELECT avg(a) FROM t15")
df = spark.sql("SELECT schema_of_avro(a) FROM t15")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.ifnull("a"), F.col("b"))
