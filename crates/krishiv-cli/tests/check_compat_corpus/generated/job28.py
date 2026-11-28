# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_25(a) FROM t28")
df = spark.sql("SELECT regexp_instr(a) FROM t28")
df = df.select(F.bit_get("a"), F.col("b"))
df = df.select(F.java_method("a"), F.col("b"))
