# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.current_timezone("a"), F.col("b"))
df = df.select(F.legacy_fn_29("a"), F.col("b"))
df = df.select(F.legacy_fn_33("a"), F.col("b"))
df = spark.sql("SELECT localtimestamp(a) FROM t12")
