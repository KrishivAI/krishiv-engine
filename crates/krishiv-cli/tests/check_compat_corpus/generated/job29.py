# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.collect_list("a"), F.col("b"))
df = spark.sql("SELECT xpath(a) FROM t29")
df = spark.sql("SELECT double(a) FROM t29")
df = spark.sql("SELECT minute(a) FROM t29")
df = df.select(F.tuple_union_theta_double("a"), F.col("b"))
df = df.select(F.legacy_fn_12("a"), F.col("b"))
