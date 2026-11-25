# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tuple_difference_double(a) FROM t24")
df = df.select(F.kll_sketch_get_rank_bigint("a"), F.col("b"))
df = spark.sql("SELECT elt(a) FROM t24")
df = df.select(F.hash("a"), F.col("b"))
