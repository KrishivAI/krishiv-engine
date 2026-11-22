# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.try_to_time("a"), F.col("b"))
df = df.select(F.kll_sketch_to_string_bigint("a"), F.col("b"))
df = df.select(F.array_agg("a"), F.col("b"))
df = spark.sql("SELECT tuple_sketch_theta_double(a) FROM t6")
df = spark.sql("SELECT xpath_short(a) FROM t6")
