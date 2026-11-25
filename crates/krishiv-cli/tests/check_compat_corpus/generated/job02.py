# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.lower("a"), F.col("b"))
df = spark.sql("SELECT sentences(a) FROM t2")
df = df.select(F.vector_sum("a"), F.col("b"))
