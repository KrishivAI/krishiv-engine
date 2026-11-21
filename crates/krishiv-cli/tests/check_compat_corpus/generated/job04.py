# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT to_avro(a) FROM t4")
df = spark.sql("SELECT sentences(a) FROM t4")
df = spark.sql("SELECT ascii(a) FROM t4")
