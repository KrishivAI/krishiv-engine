# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.sha2("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_to_string_bigint(a) FROM t24")
df = spark.sql("SELECT exp(a) FROM t24")
df = spark.sql("SELECT tuple_difference_theta_double(a) FROM t24")
