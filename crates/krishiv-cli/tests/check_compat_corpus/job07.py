# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT last(a) FROM t7")
df = spark.sql("SELECT time_to_seconds(a) FROM t7")
df = spark.sql("SELECT transform_values(a) FROM t7")
df = spark.sql("SELECT tuple_difference_theta_integer(a) FROM t7")
df = spark.sql("SELECT kll_merge_agg_bigint(a) FROM t7")
