# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_integer("a"), F.col("b"))
df = spark.sql("SELECT sec(a) FROM t20")
df = df.select(F.tuple_difference_double("a"), F.col("b"))
df = df.select(F.regr_sxx("a"), F.col("b"))
df = df.select(F.st_setsrid("a"), F.col("b"))
df = spark.sql("SELECT bitmap_construct_agg(a) FROM t20")
