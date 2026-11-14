# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_difference_theta_integer("a"), F.col("b"))
df = spark.sql("SELECT size(a) FROM t20")
df = df.select(F.kll_sketch_to_string_float("a"), F.col("b"))
df = df.select(F.schema_of_variant("a"), F.col("b"))
df = df.select(F.tan("a"), F.col("b"))
df = spark.sql("SELECT bround(a) FROM t20")
