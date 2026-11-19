# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_35(a) FROM t13")
df = spark.sql("SELECT legacy_fn_22(a) FROM t13")
df = spark.sql("SELECT xxhash64(a) FROM t13")
df = df.select(F.tuple_difference_theta_integer("a"), F.col("b"))
