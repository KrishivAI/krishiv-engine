# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = df.select(F.schema_of_avro("a"), F.col("b"))
df = spark.sql("SELECT forall(a) FROM t0")
df = df.select(F.legacy_fn_0("a"), F.col("b"))
df = spark.sql("SELECT explode_outer(a) FROM t0")
df = spark.sql("SELECT time_to_micros(a) FROM t0")
