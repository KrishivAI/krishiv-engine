# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.randstr("a"), F.col("b"))
df = spark.sql("SELECT xpath_string(a) FROM t11")
df = spark.sql("SELECT sentences(a) FROM t11")
df = spark.sql("SELECT sentences(a) FROM t11")
df = df.select(F.str_to_map("a"), F.col("b"))
df = spark.sql("SELECT approx_top_k(a) FROM t11")
