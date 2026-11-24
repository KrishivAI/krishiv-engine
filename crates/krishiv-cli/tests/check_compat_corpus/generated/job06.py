# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT make_interval(a) FROM t6")
df = spark.sql("SELECT try_subtract(a) FROM t6")
df = spark.sql("SELECT kll_sketch_get_rank_double(a) FROM t6")
df = df.select(F.input_file_block_start("a"), F.col("b"))
