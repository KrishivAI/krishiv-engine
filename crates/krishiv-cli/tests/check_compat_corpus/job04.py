# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.kll_sketch_to_string_double("a"), F.col("b"))
df = df.select(F.log1p("a"), F.col("b"))
df = df.select(F.approx_top_k("a"), F.col("b"))
df = df.select(F.legacy_fn_38("a"), F.col("b"))
