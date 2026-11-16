# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.collect_list("a"), F.col("b"))
df = spark.sql("SELECT sinh(a) FROM t4")
df = spark.sql("SELECT to_variant_object(a) FROM t4")
df = spark.sql("SELECT kll_sketch_get_rank_double(a) FROM t4")
df = spark.sql("SELECT ascii(a) FROM t4")
