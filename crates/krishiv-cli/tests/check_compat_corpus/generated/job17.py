# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT factorial(a) FROM t17")
df = df.select(F.tuple_intersection_theta_double("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
df = df.select(F.log1p("a"), F.col("b"))
df = df.select(F.json_object_keys("a"), F.col("b"))
df = df.select(F.kll_sketch_get_n_double("a"), F.col("b"))
