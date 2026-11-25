# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT count_min_sketch(a) FROM t10")
df = spark.sql("SELECT tuple_sketch_agg_integer(a) FROM t10")
df = spark.sql("SELECT legacy_fn_0(a) FROM t10")
df = df.select(F.tuple_union_agg_integer("a"), F.col("b"))
