# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.named_struct("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_4(a) FROM t11")
df = spark.sql("SELECT rlike(a) FROM t11")
