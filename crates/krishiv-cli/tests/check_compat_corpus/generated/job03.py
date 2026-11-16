# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT round(a) FROM t3")
df = spark.sql("SELECT inline(a) FROM t3")
df = spark.sql("SELECT tuple_sketch_estimate_double(a) FROM t3")
df = df.select(F.count_min_sketch("a"), F.col("b"))
