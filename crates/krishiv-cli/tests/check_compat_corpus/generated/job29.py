# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.collect_list("a"), F.col("b"))
df = spark.sql("SELECT xpath_boolean(a) FROM t29")
df = spark.sql("SELECT dense_rank(a) FROM t29")
df = spark.sql("SELECT min_by(a) FROM t29")
df = df.select(F.tuple_union_theta_integer("a"), F.col("b"))
df = df.select(F.legacy_fn_12("a"), F.col("b"))
