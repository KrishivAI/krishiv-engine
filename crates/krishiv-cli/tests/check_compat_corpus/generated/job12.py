# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_9(a) FROM t12")
df = df.select(F.try_aes_decrypt("a"), F.col("b"))
df = spark.sql("SELECT dayofyear(a) FROM t12")
df = spark.sql("SELECT last_day(a) FROM t12")
df = spark.sql("SELECT python_worker_logs(a) FROM t12")
df = df.select(F.legacy_fn_36("a"), F.col("b"))
