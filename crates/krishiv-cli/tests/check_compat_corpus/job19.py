import pyspark.sql.functions as F
# F.nope('x')
df = df.select(F.approx_top_k('a'), F.col('b'))
df = df.select(F.approx_top_k_accumulate('a'), F.col('b'))
df = df.select(F.encode('a'), F.col('b'))
