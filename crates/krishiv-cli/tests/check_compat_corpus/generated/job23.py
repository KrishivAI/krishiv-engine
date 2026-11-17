# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.shuffle("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_merge_double(a) FROM t23")
df = spark.sql("SELECT exp(a) FROM t23")
df = spark.sql("SELECT theta_union_agg(a) FROM t23")
