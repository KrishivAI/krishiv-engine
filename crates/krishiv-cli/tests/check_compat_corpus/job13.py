import pyspark.sql.functions as F
# F.nope('x')
df = spark.sql("SELECT time_from_millis(a) FROM t13")
df = df.select(F.array_except('a'), F.col('b'))
df = df.select(F.variant_get('a'), F.col('b'))
df = df.select(F.concat('a'), F.col('b'))
df = df.select(F.tuple_intersection_integer('a'), F.col('b'))
df = spark.sql("SELECT map_from_entries(a) FROM t13")
