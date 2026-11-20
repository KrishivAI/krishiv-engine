# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT split(a) FROM t13")
df = df.select(F.vector_normalize("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_9(a) FROM t13")
df = df.select(F.ceiling("a"), F.col("b"))
df = df.select(F.xpath("a"), F.col("b"))
df = spark.sql("SELECT locate(a) FROM t13")
