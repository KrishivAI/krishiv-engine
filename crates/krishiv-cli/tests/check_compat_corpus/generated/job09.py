# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT array_remove(a) FROM t9")
df = df.select(F.legacy_fn_28("a"), F.col("b"))
df = df.select(F.input_file_block_start("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_39(a) FROM t9")
df = spark.sql("SELECT legacy_fn_34(a) FROM t9")
