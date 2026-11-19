# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.sqrt("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_to_string_double(a) FROM t22")
df = df.select(F.assert_true("a"), F.col("b"))
df = spark.sql("SELECT transform_values(a) FROM t22")
df = df.select(F.nanvl("a"), F.col("b"))
