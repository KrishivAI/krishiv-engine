# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT filter(a) FROM t17")
df = df.select(F.tuple_intersection_integer("a"), F.col("b"))
df = df.select(F.legacy_fn_37("a"), F.col("b"))
df = df.select(F.log2("a"), F.col("b"))
df = df.select(F.json_tuple("a"), F.col("b"))
df = df.select(F.kll_sketch_get_n_bigint("a"), F.col("b"))
