# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT instr(a) FROM t7")
df = spark.sql("SELECT str_to_map(a) FROM t7")
df = spark.sql("SELECT vector_avg(a) FROM t7")
df = df.select(F.element_at("a"), F.col("b"))
df = df.select(F.legacy_fn_1("a"), F.col("b"))
