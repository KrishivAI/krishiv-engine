# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.split("a"), F.col("b"))
df = spark.sql("SELECT date_part(a) FROM t24")
df = df.select(F.convert_timezone("a"), F.col("b"))
df = spark.sql("SELECT try_parse_url(a) FROM t24")
df = spark.sql("SELECT cume_dist(a) FROM t24")
df = df.select(F.map("a"), F.col("b"))
