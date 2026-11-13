# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.current_time("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_get_quantile_float(a) FROM t2")
df = spark.sql("SELECT kll_sketch_get_quantile_bigint(a) FROM t2")
