# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.xpath_string("a"), F.col("b"))
df = df.select(F.rank("a"), F.col("b"))
df = df.select(F.kll_sketch_agg_bigint("a"), F.col("b"))
df = df.select(F.regr_sxx("a"), F.col("b"))
df = df.select(F.st_setsrid("a"), F.col("b"))
