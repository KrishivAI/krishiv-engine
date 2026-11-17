# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT sqrt(a) FROM t12")
df = df.select(F.weekofyear("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_9(a) FROM t12")
df = df.select(F.ceiling("a"), F.col("b"))
df = df.select(F.tuple_union_double("a"), F.col("b"))
df = spark.sql("SELECT log10(a) FROM t12")
