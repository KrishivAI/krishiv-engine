# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT theta_sketch_agg(a) FROM t27")
df = df.select(F.lead("a"), F.col("b"))
df = spark.sql("SELECT hll_union_agg(a) FROM t27")
df = spark.sql("SELECT approx_top_k_accumulate(a) FROM t27")
