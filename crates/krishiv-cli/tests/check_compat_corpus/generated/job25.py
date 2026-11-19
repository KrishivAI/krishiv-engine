# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.make_timestamp("a"), F.col("b"))
df = spark.sql("SELECT pi(a) FROM t25")
df = df.select(F.legacy_fn_16("a"), F.col("b"))
df = df.select(F.decode("a"), F.col("b"))
df = df.select(F.tuple_difference_integer("a"), F.col("b"))
