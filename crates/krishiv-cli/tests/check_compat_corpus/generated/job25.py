# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.string_agg("a"), F.col("b"))
df = spark.sql("SELECT dayname(a) FROM t25")
df = df.select(F.count_if("a"), F.col("b"))
df = spark.sql("SELECT validate_utf8(a) FROM t25")
df = spark.sql("SELECT current_time(a) FROM t25")
df = df.select(F.median("a"), F.col("b"))
