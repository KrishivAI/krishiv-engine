# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.xpath_long("a"), F.col("b"))
df = df.select(F.hash("a"), F.col("b"))
df = df.select(F.sqrt("a"), F.col("b"))
df = df.select(F.legacy_fn_8("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_get_n_float(a) FROM t20")
