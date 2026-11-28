# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT reflect(a) FROM t19")
df = spark.sql("SELECT date_from_unix_date(a) FROM t19")
df = df.select(F.legacy_fn_24("a"), F.col("b"))
df = df.select(F.ilike("a"), F.col("b"))
