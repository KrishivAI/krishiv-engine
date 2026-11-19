# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT startswith(a) FROM t13")
df = df.select(F.zeroifnull("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_9(a) FROM t13")
df = df.select(F.collate("a"), F.col("b"))
df = df.select(F.tuple_sketch_summary_double("a"), F.col("b"))
df = spark.sql("SELECT make_dt_interval(a) FROM t13")
