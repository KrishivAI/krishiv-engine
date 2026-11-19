# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.kll_sketch_agg_double("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_9(a) FROM t3")
df = df.select(F.map_filter("a"), F.col("b"))
