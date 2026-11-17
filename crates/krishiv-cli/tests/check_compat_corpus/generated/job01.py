# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_char(a) FROM t1")
df = df.select(F.mask("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_get_rank_float(a) FROM t1")
