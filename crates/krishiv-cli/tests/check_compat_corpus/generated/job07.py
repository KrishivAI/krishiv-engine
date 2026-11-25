# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT get_json_object(a) FROM t7")
df = spark.sql("SELECT sin(a) FROM t7")
df = df.select(F.json_tuple("a"), F.col("b"))
df = spark.sql("SELECT java_method(a) FROM t7")
df = df.select(F.legacy_fn_35("a"), F.col("b"))
