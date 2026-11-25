# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.char_length("a"), F.col("b"))
df = spark.sql("SELECT regr_intercept(a) FROM t4")
df = spark.sql("SELECT variant_get(a) FROM t4")
df = df.select(F.parse_json("a"), F.col("b"))
df = df.select(F.tuple_difference_integer("a"), F.col("b"))
