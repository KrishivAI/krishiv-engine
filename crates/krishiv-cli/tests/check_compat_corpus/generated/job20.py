# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.tuple_sketch_agg_double("a"), F.col("b"))
df = spark.sql("SELECT shuffle(a) FROM t20")
df = df.select(F.schema_of_avro("a"), F.col("b"))
df = df.select(F.rpad("a"), F.col("b"))
df = df.select(F.struct("a"), F.col("b"))
df = spark.sql("SELECT bround(a) FROM t20")
