# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.split_part("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_to_string_float(a) FROM t23")
df = df.select(F.assert_true("a"), F.col("b"))
df = spark.sql("SELECT to_xml(a) FROM t23")
df = df.select(F.months_between("a"), F.col("b"))
