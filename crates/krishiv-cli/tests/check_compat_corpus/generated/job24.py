# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.split_part("a"), F.col("b"))
df = spark.sql("SELECT date_sub(a) FROM t24")
df = df.select(F.convert_timezone("a"), F.col("b"))
df = spark.sql("SELECT try_subtract(a) FROM t24")
df = spark.sql("SELECT curdate(a) FROM t24")
df = df.select(F.map_concat("a"), F.col("b"))
