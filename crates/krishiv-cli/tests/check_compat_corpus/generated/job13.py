# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT session_user(a) FROM t13")
df = df.select(F.try_validate_utf8("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_9(a) FROM t13")
df = df.select(F.ceiling("a"), F.col("b"))
df = df.select(F.tuple_union_double("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_merge_bigint(a) FROM t13")
