# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT split_part(a) FROM t12")
df = df.select(F.version("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_9(a) FROM t12")
df = df.select(F.ceiling("a"), F.col("b"))
df = df.select(F.tuple_union_theta_double("a"), F.col("b"))
df = spark.sql("SELECT log(a) FROM t12")
