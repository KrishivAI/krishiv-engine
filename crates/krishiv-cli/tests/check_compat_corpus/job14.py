import pyspark.sql.functions as F
# F.nope('x')
df = spark.sql("SELECT legacy_fn_35(a) FROM t14")
df = spark.sql("SELECT legacy_fn_22(a) FROM t14")
df = spark.sql("SELECT regr_syy(a) FROM t14")
df = df.select(F.sentences('a'), F.col('b'))
