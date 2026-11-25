# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_24(a) FROM t16")
df = spark.sql("SELECT nanvl(a) FROM t16")
df = spark.sql("SELECT tuple_sketch_summary_integer(a) FROM t16")
df = spark.sql("SELECT legacy_fn_21(a) FROM t16")
df = spark.sql("SELECT tuple_sketch_agg_double(a) FROM t16")
df = spark.sql("SELECT kll_sketch_get_n_float(a) FROM t16")
