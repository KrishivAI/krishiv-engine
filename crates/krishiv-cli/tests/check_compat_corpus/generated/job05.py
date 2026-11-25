# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.hash("a"), F.col("b"))
df = spark.sql("SELECT tuple_intersection_theta_double(a) FROM t5")
df = df.select(F.legacy_fn_12("a"), F.col("b"))
