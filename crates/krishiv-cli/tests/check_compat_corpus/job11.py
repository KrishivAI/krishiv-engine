import pyspark.sql.functions as F
# F.nope('x')
df = df.select(F.regexp_replace('a'), F.col('b'))
df = spark.sql("SELECT legacy_fn_4(a) FROM t11")
df = spark.sql("SELECT str_to_map(a) FROM t11")
