# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.to_time("a"), F.col("b"))
df = spark.sql("SELECT any_value(a) FROM t0")
df = df.select(F.st_geomfromwkb("a"), F.col("b"))
df = df.select(F.xpath_float("a"), F.col("b"))
df = spark.sql("SELECT regexp_extract_all(a) FROM t0")
