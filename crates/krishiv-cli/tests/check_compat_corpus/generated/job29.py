# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.collect_list("a"), F.col("b"))
df = spark.sql("SELECT xpath_double(a) FROM t29")
df = spark.sql("SELECT legacy_fn_29(a) FROM t29")
df = spark.sql("SELECT double(a) FROM t29")
df = spark.sql("SELECT monotonically_increasing_id(a) FROM t29")
df = df.select(F.tuple_sketch_estimate_double("a"), F.col("b"))
