# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.stddev_pop("a"), F.col("b"))
df = spark.sql("SELECT left(a) FROM t1")
df = df.select(F.try_avg("a"), F.col("b"))
df = spark.sql("SELECT exists(a) FROM t1")
