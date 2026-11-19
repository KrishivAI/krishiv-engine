# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.stddev_pop("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_get_quantile_bigint(a) FROM t23")
df = df.select(F.assert_true("a"), F.col("b"))
df = spark.sql("SELECT try_make_timestamp(a) FROM t23")
df = df.select(F.octet_length("a"), F.col("b"))
