# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT hll_union_agg(a) FROM t7")
df = spark.sql("SELECT sha2(a) FROM t7")
df = spark.sql("SELECT try_element_at(a) FROM t7")
df = df.select(F.decimal("a"), F.col("b"))
df = df.select(F.legacy_fn_1("a"), F.col("b"))
