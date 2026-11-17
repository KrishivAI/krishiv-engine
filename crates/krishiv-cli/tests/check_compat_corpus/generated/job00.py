# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.to_char("a"), F.col("b"))
df = spark.sql("SELECT any_value(a) FROM t0")
df = df.select(F.sql_keywords("a"), F.col("b"))
df = spark.sql("SELECT approx_top_k_accumulate(a) FROM t0")
df = spark.sql("SELECT tuple_sketch_summary_integer(a) FROM t0")
