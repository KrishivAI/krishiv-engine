# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_sketch_agg_double("a"), F.col("b"))
df = spark.sql("SELECT theta_sketch_estimate(a) FROM t9")
df = df.select(F.regexp_instr("a"), F.col("b"))
df = df.select(F.tuple_union_agg_double("a"), F.col("b"))
df = df.select(F.legacy_fn_35("a"), F.col("b"))
