# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_merge_agg_float(a) FROM t3")
df = spark.sql("SELECT lead(a) FROM t3")
df = spark.sql("SELECT date_sub(a) FROM t3")
df = df.select(F.approx_top_k_combine("a"), F.col("b"))
df = spark.sql("SELECT regexp_substr(a) FROM t3")
df = df.select(F.xpath_number("a"), F.col("b"))
