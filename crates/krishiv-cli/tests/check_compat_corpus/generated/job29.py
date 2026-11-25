# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_25(a) FROM t29")
df = spark.sql("SELECT position(a) FROM t29")
df = df.select(F.bit_get("a"), F.col("b"))
df = df.select(F.approx_top_k_combine("a"), F.col("b"))
