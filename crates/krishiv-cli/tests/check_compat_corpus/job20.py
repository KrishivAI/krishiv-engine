import pyspark.sql.functions as F
# F.nope('x')
df = df.select(F.tuple_difference_theta_integer('a'), F.col('b'))
df = spark.sql("SELECT sqrt(a) FROM t20")
df = df.select(F.kll_sketch_to_string_float('a'), F.col('b'))
df = df.select(F.shiftrightunsigned('a'), F.col('b'))
df = df.select(F.timestamp_seconds('a'), F.col('b'))
df = spark.sql("SELECT cbrt(a) FROM t20")
