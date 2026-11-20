# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.approx_top_k("a"), F.col("b"))
df = spark.sql("SELECT raise_error(a) FROM t15")
df = spark.sql("SELECT xpath_number(a) FROM t15")
