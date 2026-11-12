import pyspark.sql.functions as F
# F.nope('x')
df = df.select(F.time_to_micros('a'), F.col('b'))
df = df.select(F.xpath_int('a'), F.col('b'))
df = df.select(F.time_from_seconds('a'), F.col('b'))
df = df.select(F.legacy_fn_24('a'), F.col('b'))
df = df.select(F.time_trunc('a'), F.col('b'))
