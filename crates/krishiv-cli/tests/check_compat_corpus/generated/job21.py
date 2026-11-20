# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.dayofmonth("a"), F.col("b"))
df = spark.sql("SELECT java_method(a) FROM t21")
df = spark.sql("SELECT element_at(a) FROM t21")
df = spark.sql("SELECT kll_merge_agg_bigint(a) FROM t21")
df = df.select(F.sentences("a"), F.col("b"))
