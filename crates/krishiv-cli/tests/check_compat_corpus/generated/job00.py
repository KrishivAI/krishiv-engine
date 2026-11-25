# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.e("a"), F.col("b"))
df = df.select(F.approx_top_k("a"), F.col("b"))
df = spark.sql("SELECT json_array_length(a) FROM t0")
df = spark.sql("SELECT sort_array(a) FROM t0")
df = spark.sql("SELECT kll_sketch_merge_double(a) FROM t0")
