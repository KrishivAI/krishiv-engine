# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_sketch_to_string_float(a) FROM t23")
df = df.select(F.levenshtein("a"), F.col("b"))
df = spark.sql("SELECT element_at(a) FROM t23")
df = df.select(F.grouping_id("a"), F.col("b"))
