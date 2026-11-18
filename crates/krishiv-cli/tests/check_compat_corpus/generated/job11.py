# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_9(a) FROM t11")
df = df.select(F.to_xml("a"), F.col("b"))
df = spark.sql("SELECT datediff(a) FROM t11")
df = spark.sql("SELECT isnan(a) FROM t11")
df = spark.sql("SELECT theta_sketch_estimate(a) FROM t11")
df = df.select(F.legacy_fn_36("a"), F.col("b"))
