import pyspark.sql.functions as F
# F.nope('x')
df = spark.sql("SELECT kll_sketch_agg_float(a) FROM t24")
df = df.select(F.map_entries('a'), F.col('b'))
df = spark.sql("SELECT find_in_set(a) FROM t24")
df = df.select(F.isnull('a'), F.col('b'))
