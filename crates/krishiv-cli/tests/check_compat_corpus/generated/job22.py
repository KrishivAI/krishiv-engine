# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.st_geogfromwkb("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_to_string_bigint(a) FROM t22")
df = df.select(F.assert_true("a"), F.col("b"))
df = spark.sql("SELECT trim(a) FROM t22")
df = df.select(F.negative("a"), F.col("b"))
