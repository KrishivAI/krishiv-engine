# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tuple_sketch_summary_double(a) FROM t0")
df = df.select(F.legacy_fn_8("a"), F.col("b"))
df = spark.sql("SELECT python_worker_logs(a) FROM t0")
