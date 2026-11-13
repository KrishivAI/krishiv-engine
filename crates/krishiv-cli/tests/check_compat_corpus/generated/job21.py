# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.dense_rank("a"), F.col("b"))
df = spark.sql("SELECT bitmap_bucket_number(a) FROM t21")
df = spark.sql("SELECT exists(a) FROM t21")
df = spark.sql("SELECT bitmap_construct_agg(a) FROM t21")
df = df.select(F.kll_sketch_get_rank_bigint("a"), F.col("b"))
