# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.slice("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_get_n_float(a) FROM t24")
df = spark.sql("SELECT filter(a) FROM t24")
df = spark.sql("SELECT sentences(a) FROM t24")
