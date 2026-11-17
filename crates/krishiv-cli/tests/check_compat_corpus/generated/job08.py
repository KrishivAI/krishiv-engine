# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_sketch_theta_double("a"), F.col("b"))
df = spark.sql("SELECT tuple_difference_theta_double(a) FROM t8")
df = df.select(F.randstr("a"), F.col("b"))
df = df.select(F.window("a"), F.col("b"))
df = df.select(F.legacy_fn_35("a"), F.col("b"))
