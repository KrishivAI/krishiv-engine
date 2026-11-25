# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.current_user("a"), F.col("b"))
df = df.select(F.legacy_fn_29("a"), F.col("b"))
df = df.select(F.legacy_fn_33("a"), F.col("b"))
df = spark.sql("SELECT locate(a) FROM t12")
