# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.reflect("a"), F.col("b"))
df = spark.sql("SELECT substring_index(a) FROM t29")
df = spark.sql("SELECT unix_millis(a) FROM t29")
df = spark.sql("SELECT xpath_int(a) FROM t29")
