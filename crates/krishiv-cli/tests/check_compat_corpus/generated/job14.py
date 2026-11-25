# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_35(a) FROM t14")
df = spark.sql("SELECT legacy_fn_22(a) FROM t14")
df = spark.sql("SELECT var_pop(a) FROM t14")
df = spark.sql("SELECT xpath_long(a) FROM t14")
