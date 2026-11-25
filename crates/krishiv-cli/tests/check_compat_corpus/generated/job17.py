# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_24(a) FROM t17")
df = spark.sql("SELECT map_from_entries(a) FROM t17")
df = spark.sql("SELECT tuple_sketch_agg_integer(a) FROM t17")
df = spark.sql("SELECT legacy_fn_21(a) FROM t17")
df = spark.sql("SELECT tuple_intersection_theta_double(a) FROM t17")
df = spark.sql("SELECT python_worker_logs(a) FROM t17")
