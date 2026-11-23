# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_sketch_summary_integer("a"), F.col("b"))
df = spark.sql("SELECT sentences(a) FROM t15")
df = df.select(F.array_except("a"), F.col("b"))
df = df.select(F.hypot("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_22(a) FROM t15")
