import pyspark.sql.functions as F
# F.nope('x')
df = df.select(F.approx_top_k('a'), F.col('b'))
df = spark.sql("SELECT regr_syy(a) FROM t15")
df = spark.sql("SELECT tuple_sketch_estimate_integer(a) FROM t15")
