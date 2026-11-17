# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.approx_top_k("a"), F.col("b"))
df = spark.sql("SELECT randn(a) FROM t14")
df = spark.sql("SELECT xpath(a) FROM t14")
