# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT xpath_long(a) FROM t1")
df = spark.sql("SELECT legacy_fn_7(a) FROM t1")
df = df.select(F.dayofmonth("a"), F.col("b"))
df = df.select(F.tuple_difference_theta_integer("a"), F.col("b"))
