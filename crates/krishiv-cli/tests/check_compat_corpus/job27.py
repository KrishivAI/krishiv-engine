import pyspark.sql.functions as F
# F.nope('x')
df = spark.sql("SELECT try_url_decode(a) FROM t27")
df = df.select(F.minute('a'), F.col('b'))
df = spark.sql("SELECT last(a) FROM t27")
df = spark.sql("SELECT approx_top_k_estimate(a) FROM t27")
