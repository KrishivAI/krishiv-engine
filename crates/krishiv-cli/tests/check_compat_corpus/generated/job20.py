# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.dayofmonth("a"), F.col("b"))
df = spark.sql("SELECT from_protobuf(a) FROM t20")
df = spark.sql("SELECT element_at(a) FROM t20")
df = spark.sql("SELECT histogram_numeric(a) FROM t20")
df = df.select(F.schema_of_avro("a"), F.col("b"))
