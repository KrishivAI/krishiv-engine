# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_sketch_agg_bigint(a) FROM t22")
df = df.select(F.date("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_35(a) FROM t22")
