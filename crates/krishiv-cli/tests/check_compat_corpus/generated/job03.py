# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_binary(a) FROM t3")
df = df.select(F.user("a"), F.col("b"))
df = df.select(F.bit_and("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_20(a) FROM t3")
