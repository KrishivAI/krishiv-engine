# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT try_element_at(a) FROM t27")
df = df.select(F.map_values("a"), F.col("b"))
df = spark.sql("SELECT is_valid_utf8(a) FROM t27")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t27")
