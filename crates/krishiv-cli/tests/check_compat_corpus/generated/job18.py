# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT kll_sketch_merge_bigint(a) FROM t18")
df = spark.sql("SELECT bigint(a) FROM t18")
df = df.select(F.kll_sketch_agg_double("a"), F.col("b"))
df = df.select(F.regr_slope("a"), F.col("b"))
df = df.select(F.st_geomfromwkb("a"), F.col("b"))
