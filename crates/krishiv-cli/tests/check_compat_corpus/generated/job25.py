# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT reflect(a) FROM t25")
df = df.select(F.reflect("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_35(a) FROM t25")
df = spark.sql("SELECT legacy_fn_3(a) FROM t25")
df = spark.sql("SELECT legacy_fn_5(a) FROM t25")
df = spark.sql("SELECT approx_percentile(a) FROM t25")
