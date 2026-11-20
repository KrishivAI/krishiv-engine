# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tuple_difference_double(a) FROM t2")
df = spark.sql("SELECT date(a) FROM t2")
df = df.select(F.approx_top_k_combine("a"), F.col("b"))
