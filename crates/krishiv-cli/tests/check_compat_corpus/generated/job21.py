# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT xpath_double(a) FROM t21")
df = spark.sql("SELECT array_compact(a) FROM t21")
df = df.select(F.sign("a"), F.col("b"))
