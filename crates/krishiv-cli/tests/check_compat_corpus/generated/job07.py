# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_28("a"), F.col("b"))
df = df.select(F.shiftrightunsigned("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_39(a) FROM t7")
df = spark.sql("SELECT legacy_fn_34(a) FROM t7")
