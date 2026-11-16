# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT int(a) FROM t7")
df = spark.sql("SELECT string_agg(a) FROM t7")
df = spark.sql("SELECT vector_l2_distance(a) FROM t7")
df = df.select(F.element_at("a"), F.col("b"))
df = df.select(F.legacy_fn_1("a"), F.col("b"))
