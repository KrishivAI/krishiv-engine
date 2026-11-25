# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.sha("a"), F.col("b"))
df = spark.sql("SELECT date_sub(a) FROM t25")
df = df.select(F.convert_timezone("a"), F.col("b"))
df = spark.sql("SELECT to_variant_object(a) FROM t25")
df = spark.sql("SELECT curdate(a) FROM t25")
df = df.select(F.ln("a"), F.col("b"))
