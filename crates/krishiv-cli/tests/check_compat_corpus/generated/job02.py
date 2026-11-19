# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.minute("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_get_n_bigint(a) FROM t2")
df = spark.sql("SELECT last(a) FROM t2")
