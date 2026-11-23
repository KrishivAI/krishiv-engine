# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_sketch_to_string_double(a) FROM t23")
df = df.select(F.listagg("a"), F.col("b"))
df = spark.sql("SELECT elt(a) FROM t23")
df = df.select(F.hash("a"), F.col("b"))
