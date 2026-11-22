# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_7(a) FROM t2")
df = df.select(F.dayofweek("a"), F.col("b"))
df = df.select(F.theta_sketch_agg("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_33(a) FROM t2")
