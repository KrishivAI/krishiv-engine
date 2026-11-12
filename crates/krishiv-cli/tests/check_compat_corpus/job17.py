# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT theta_sketch_agg(a) FROM t17")
df = df.select(F.expm1("a"), F.col("b"))
df = spark.sql("SELECT get(a) FROM t17")
df = spark.sql("SELECT arrays_overlap(a) FROM t17")
df = spark.sql("SELECT try_url_decode(a) FROM t17")
