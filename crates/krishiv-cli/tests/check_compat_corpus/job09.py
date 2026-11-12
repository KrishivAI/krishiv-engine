import pyspark.sql.functions as F
# F.nope('x')
df = df.select(F.tuple_difference_theta_integer('a'), F.col('b'))
df = spark.sql("SELECT session_window(a) FROM t9")
df = df.select(F.replace('a'), F.col('b'))
df = df.select(F.tuple_sketch_agg_integer('a'), F.col('b'))
df = df.select(F.legacy_fn_35('a'), F.col('b'))
