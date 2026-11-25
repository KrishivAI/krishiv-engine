# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT rpad(a) FROM t4")
df = df.select(F.greatest("a"), F.col("b"))
df = spark.sql("SELECT python_worker_logs(a) FROM t4")
df = df.select(F.array_size("a"), F.col("b"))
df = df.select(F.python_worker_logs("a"), F.col("b"))
