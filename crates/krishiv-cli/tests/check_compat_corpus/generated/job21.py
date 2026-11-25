# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_35(a) FROM t21")
df = df.select(F.legacy_fn_12("a"), F.col("b"))
df = df.select(F.regr_slope("a"), F.col("b"))
