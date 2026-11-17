# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_merge_agg_double(a) FROM t21")
df = df.select(F.date("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_35(a) FROM t21")
