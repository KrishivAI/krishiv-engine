# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT printf(a) FROM t3")
df = spark.sql("SELECT hash(a) FROM t3")
df = spark.sql("SELECT tuple_union_integer(a) FROM t3")
df = df.select(F.count_min_sketch("a"), F.col("b"))
