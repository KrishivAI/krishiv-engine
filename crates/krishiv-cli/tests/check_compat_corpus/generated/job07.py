# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT input_file_name(a) FROM t7")
df = spark.sql("SELECT stddev_samp(a) FROM t7")
df = spark.sql("SELECT variant_explode_outer(a) FROM t7")
df = df.select(F.element_at("a"), F.col("b"))
df = df.select(F.legacy_fn_1("a"), F.col("b"))
