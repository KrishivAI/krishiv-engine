# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.posexplode_outer("a"), F.col("b"))
df = spark.sql("SELECT legacy_fn_4(a) FROM t10")
df = spark.sql("SELECT sin(a) FROM t10")
