# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT right(a) FROM t3")
df = spark.sql("SELECT ilike(a) FROM t3")
df = spark.sql("SELECT tuple_sketch_theta_integer(a) FROM t3")
df = df.select(F.count_min_sketch("a"), F.col("b"))
