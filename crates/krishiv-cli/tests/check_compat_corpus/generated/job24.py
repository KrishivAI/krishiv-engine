# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.bit_and("a"), F.col("b"))
df = df.select(F.from_csv("a"), F.col("b"))
df = spark.sql("SELECT xpath_int(a) FROM t24")
