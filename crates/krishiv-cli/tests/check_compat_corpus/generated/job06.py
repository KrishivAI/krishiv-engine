# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT hour(a) FROM t6")
df = spark.sql("SELECT st_geogfromwkb(a) FROM t6")
df = spark.sql("SELECT url_encode(a) FROM t6")
df = df.select(F.decimal("a"), F.col("b"))
df = df.select(F.legacy_fn_1("a"), F.col("b"))
