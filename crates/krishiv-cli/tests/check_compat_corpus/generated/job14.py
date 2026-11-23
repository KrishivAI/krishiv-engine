# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_35(a) FROM t14")
df = spark.sql("SELECT legacy_fn_22(a) FROM t14")
df = spark.sql("SELECT width_bucket(a) FROM t14")
df = df.select(F.tuple_intersection_double("a"), F.col("b"))
