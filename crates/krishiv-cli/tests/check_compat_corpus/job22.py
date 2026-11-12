# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT collations(a) FROM t22")
df = df.select(F.day("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_35(a) FROM t22")
