# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT size(a) FROM t28")
df = df.select(F.try_parse_json("a"), F.col("b"))
df = df.select(F.legacy_fn_33("a"), F.col("b"))
