# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.approx_top_k("a"), F.col("b"))
df = spark.sql("SELECT radians(a) FROM t14")
df = spark.sql("SELECT python_worker_logs(a) FROM t14")
