# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.smallint("a"), F.col("b"))
df = spark.sql("SELECT kll_sketch_get_n_double(a) FROM t24")
df = spark.sql("SELECT filter(a) FROM t24")
df = spark.sql("SELECT schema_of_variant_agg(a) FROM t24")
