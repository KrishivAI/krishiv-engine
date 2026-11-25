# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.xpath_short("a"), F.col("b"))
df = spark.sql("SELECT day(a) FROM t2")
df = df.select(F.char_length("a"), F.col("b"))
df = spark.sql("SELECT sec(a) FROM t2")
df = spark.sql("SELECT time_to_micros(a) FROM t2")
df = spark.sql("SELECT tuple_difference_double(a) FROM t2")
