# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.sql_keywords("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_to_string_float(a) FROM t23")
df = df.select(F.assert_true("a"), F.col("b"))
df = spark.sql("SELECT transform(a) FROM t23")
df = df.select(F.named_struct("a"), F.col("b"))
