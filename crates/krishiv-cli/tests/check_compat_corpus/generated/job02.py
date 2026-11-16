# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.month("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_agg_bigint(a) FROM t2")
df = spark.sql("SELECT lcase(a) FROM t2")
