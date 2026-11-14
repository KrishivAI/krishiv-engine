# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.randstr("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_4(a) FROM t11")
df = spark.sql("SELECT sqrt(a) FROM t11")
