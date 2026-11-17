# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_9(a) FROM t11")
df = df.select(F.transform_keys("a"), F.col("b"))
df = spark.sql("SELECT datediff(a) FROM t11")
df = spark.sql("SELECT isnotnull(a) FROM t11")
df = spark.sql("SELECT theta_intersection_agg(a) FROM t11")
df = df.select(F.legacy_fn_36("a"), F.col("b"))
