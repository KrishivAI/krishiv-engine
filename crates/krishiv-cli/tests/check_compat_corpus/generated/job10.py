# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT java_method(a) FROM t10")
df = spark.sql("SELECT tuple_sketch_estimate_double(a) FROM t10")
df = spark.sql("SELECT legacy_fn_0(a) FROM t10")
df = df.select(F.tuple_union_double("a"), F.col("b"))
