# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_difference_double("a"), F.col("b"))
df = df.select(F.legacy_fn_12("a"), F.col("b"))
df = df.select(F.tuple_difference_theta_integer("a"), F.col("b"))
