# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT legacy_fn_9(a) FROM t12")
df = spark.sql("SELECT vector_cosine_similarity(a) FROM t12")
df = spark.sql("SELECT legacy_fn_13(a) FROM t12")
df = spark.sql("SELECT is_variant_null(a) FROM t12")
df = spark.sql("SELECT tuple_intersection_double(a) FROM t12")
df = df.select(F.legacy_fn_36("a"), F.col("b"))
