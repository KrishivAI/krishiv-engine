# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT hll_union_agg(a) FROM t7")
df = spark.sql("SELECT sqrt(a) FROM t7")
df = spark.sql("SELECT unix_timestamp(a) FROM t7")
df = df.select(F.decimal("a"), F.col("b"))
df = df.select(F.legacy_fn_1("a"), F.col("b"))
