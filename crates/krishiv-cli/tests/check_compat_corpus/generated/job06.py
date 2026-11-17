# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT hypot(a) FROM t6")
df = spark.sql("SELECT st_setsrid(a) FROM t6")
df = spark.sql("SELECT uuid(a) FROM t6")
df = df.select(F.decimal("a"), F.col("b"))
df = df.select(F.legacy_fn_1("a"), F.col("b"))
