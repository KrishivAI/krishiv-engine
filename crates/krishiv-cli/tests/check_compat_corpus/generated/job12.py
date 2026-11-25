# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT btrim(a) FROM t12")
df = spark.sql("SELECT legacy_fn_3(a) FROM t12")
df = spark.sql("SELECT pi(a) FROM t12")
