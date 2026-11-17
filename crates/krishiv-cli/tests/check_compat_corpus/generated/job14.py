# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_35(a) FROM t14")
df = spark.sql("SELECT legacy_fn_22(a) FROM t14")
df = spark.sql("SELECT regexp_extract_all(a) FROM t14")
df = df.select(F.theta_intersection_agg("a"), F.col("b"))
