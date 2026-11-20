# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT variant_explode(a) FROM t0")
df = spark.sql("SELECT xpath_long(a) FROM t0")
df = df.select(F.legacy_fn_8("a"), F.col("b"))
df = spark.sql("SELECT tuple_difference_double(a) FROM t0")
df = df.select(F.reflect("a"), F.col("b"))
