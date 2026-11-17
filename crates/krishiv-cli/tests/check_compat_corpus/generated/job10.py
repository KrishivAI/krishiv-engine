# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.position("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_4(a) FROM t10")
df = spark.sql("SELECT size(a) FROM t10")
