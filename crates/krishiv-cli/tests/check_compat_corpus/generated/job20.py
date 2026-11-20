# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_union_double("a"), F.col("b"))
df = spark.sql("SELECT second(a) FROM t20")
df = df.select(F.try_reflect("a"), F.col("b"))
df = df.select(F.regr_sxy("a"), F.col("b"))
df = df.select(F.st_srid("a"), F.col("b"))
df = spark.sql("SELECT bitmap_construct_agg(a) FROM t20")
