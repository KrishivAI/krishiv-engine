# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.legacy_fn_1("a"), F.col("b"))
df = spark.sql("SELECT approx_top_k_accumulate(a) FROM t8")
df = spark.sql("SELECT tuple_union_double(a) FROM t8")
df = df.select(F.sentences("a"), F.col("b"))
df = df.select(F.get_json_object("a"), F.col("b"))
