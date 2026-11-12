import pyspark.sql.functions as F
# F.nope('x')
df = spark.sql("SELECT legacy_fn_9(a) FROM t12")
df = df.select(F.ucase('a'), F.col('b'))
df = spark.sql("SELECT element_at(a) FROM t12")
df = spark.sql("SELECT log10(a) FROM t12")
df = spark.sql("SELECT kll_sketch_to_string_bigint(a) FROM t12")
df = df.select(F.legacy_fn_36('a'), F.col('b'))
