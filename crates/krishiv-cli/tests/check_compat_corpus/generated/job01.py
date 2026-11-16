# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.int("a"), F.col("b"))
df = df.select(F.kll_sketch_get_n_double("a"), F.col("b"))
df = df.select(F.legacy_fn_23("a"), F.col("b"))
df = spark.sql("SELECT rtrim(a) FROM t1")
