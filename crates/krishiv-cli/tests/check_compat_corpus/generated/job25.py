# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.make_time("a"), F.col("b"))
df = spark.sql("SELECT percentile_disc(a) FROM t25")
df = df.select(F.legacy_fn_16("a"), F.col("b"))
df = df.select(F.decode("a"), F.col("b"))
df = df.select(F.tuple_difference_theta_integer("a"), F.col("b"))
