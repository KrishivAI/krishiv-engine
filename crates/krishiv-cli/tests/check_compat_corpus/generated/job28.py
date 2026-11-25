# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT xpath_long(a) FROM t28")
df = df.select(F.legacy_fn_19("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_33(a) FROM t28")
df = df.select(F.current_date("a"), F.col("b"))
