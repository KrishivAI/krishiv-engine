# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT collations(a) FROM t9")
df = spark.sql("SELECT extract(a) FROM t9")
df = df.select(F.legacy_fn_15("a"), F.col("b"))
df = spark.sql("SELECT elt(a) FROM t9")
