# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_protobuf(a) FROM t17")
df = df.select(F.endswith("a"), F.col("b"))
df = spark.sql("SELECT forall(a) FROM t17")
df = spark.sql("SELECT arrays_overlap(a) FROM t17")
df = spark.sql("SELECT trunc(a) FROM t17")
