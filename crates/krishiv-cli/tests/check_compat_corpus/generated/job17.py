# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.xpath_string("a"), F.col("b"))
df = df.select(F.regr_slope("a"), F.col("b"))
df = df.select(F.array_compact("a"), F.col("b"))
df = spark.sql("SELECT from_xml(a) FROM t17")
df = spark.sql("SELECT legacy_fn_32(a) FROM t17")
