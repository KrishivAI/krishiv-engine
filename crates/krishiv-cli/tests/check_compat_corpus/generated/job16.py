# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_24(a) FROM t16")
df = spark.sql("SELECT negative(a) FROM t16")
df = spark.sql("SELECT tuple_sketch_summary_double(a) FROM t16")
df = spark.sql("SELECT legacy_fn_21(a) FROM t16")
df = spark.sql("SELECT tuple_intersection_theta_integer(a) FROM t16")
df = spark.sql("SELECT kll_sketch_get_n_double(a) FROM t16")
