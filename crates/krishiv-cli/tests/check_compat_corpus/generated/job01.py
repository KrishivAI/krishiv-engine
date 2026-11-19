# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.input_file_name("a"), F.col("b"))
df = df.select(F.kll_sketch_get_quantile_double("a"), F.col("b"))
df = df.select(F.legacy_fn_23("a"), F.col("b"))
df = spark.sql("SELECT round(a) FROM t1")
