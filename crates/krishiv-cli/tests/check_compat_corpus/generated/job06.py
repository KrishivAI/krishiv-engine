# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT make_dt_interval(a) FROM t6")
df = spark.sql("SELECT try_parse_url(a) FROM t6")
df = spark.sql("SELECT kll_sketch_get_rank_float(a) FROM t6")
df = df.select(F.input_file_block_length("a"), F.col("b"))
