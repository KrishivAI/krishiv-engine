# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT reflect(a) FROM t2")
df = spark.sql("SELECT ceil(a) FROM t2")
df = spark.sql("SELECT reflect(a) FROM t2")
df = df.select(F.crc32("a"), F.col("b"))
