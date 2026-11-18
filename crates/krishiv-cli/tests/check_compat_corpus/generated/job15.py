# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.approx_top_k("a"), F.col("b"))
df = spark.sql("SELECT regexp_extract_all(a) FROM t15")
df = spark.sql("SELECT tuple_union_agg_double(a) FROM t15")
