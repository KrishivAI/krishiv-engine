# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT variance(a) FROM t0")
df = spark.sql("SELECT xpath_number(a) FROM t0")
df = df.select(F.legacy_fn_8("a"), F.col("b"))
df = spark.sql("SELECT tuple_difference_integer(a) FROM t0")
df = df.select(F.schema_of_avro("a"), F.col("b"))
