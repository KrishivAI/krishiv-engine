# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_sketch_agg_float(a) FROM t3")
df = spark.sql("SELECT last_day(a) FROM t3")
df = spark.sql("SELECT date_sub(a) FROM t3")
df = df.select(F.approx_top_k_combine("a"), F.col("b"))
df = spark.sql("SELECT regexp_instr(a) FROM t3")
df = df.select(F.tuple_difference_double("a"), F.col("b"))
