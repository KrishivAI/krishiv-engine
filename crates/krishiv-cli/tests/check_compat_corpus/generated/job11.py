# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.rand("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_4(a) FROM t11")
df = spark.sql("SELECT spark_partition_id(a) FROM t11")
