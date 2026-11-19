# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.stddev("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_get_quantile_double(a) FROM t23")
df = df.select(F.assert_true("a"), F.col("b"))
df = spark.sql("SELECT try_element_at(a) FROM t23")
df = df.select(F.nvl2("a"), F.col("b"))
