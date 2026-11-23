# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.xpath_number("a"), F.col("b"))
df = spark.sql("SELECT dayname(a) FROM t2")
df = df.select(F.char_length("a"), F.col("b"))
df = spark.sql("SELECT second(a) FROM t2")
df = spark.sql("SELECT time_to_millis(a) FROM t2")
df = spark.sql("SELECT try_reflect(a) FROM t2")
