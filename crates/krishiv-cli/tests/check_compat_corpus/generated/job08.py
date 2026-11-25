# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT bit_length(a) FROM t8")
df = df.select(F.bit_count("a"), F.col("b"))
df = df.select(F.java_method("a"), F.col("b"))
