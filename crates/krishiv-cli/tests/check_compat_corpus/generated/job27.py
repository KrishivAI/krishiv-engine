# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT translate(a) FROM t27")
df = df.select(F.map_filter("a"), F.col("b"))
df = spark.sql("SELECT input_file_name(a) FROM t27")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t27")
