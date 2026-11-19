# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT xpath_boolean(a) FROM t3")
df = df.select(F.kll_sketch_agg_bigint("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_9(a) FROM t3")
df = df.select(F.map_from_arrays("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_2(a) FROM t3")
df = spark.sql("SELECT kll_sketch_to_string_double(a) FROM t3")
