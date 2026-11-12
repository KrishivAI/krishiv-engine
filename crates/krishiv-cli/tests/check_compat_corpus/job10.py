import pyspark.sql.functions as F
# F.nope('x')
df = spark.sql("SELECT array_remove(a) FROM t10")
df = df.select(F.legacy_fn_28('a'), F.col('b'))
df = df.select(F.length('a'), F.col('b'))
df = spark.sql("SELECT legacy_fn_39(a) FROM t10")
df = spark.sql("SELECT legacy_fn_34(a) FROM t10")
