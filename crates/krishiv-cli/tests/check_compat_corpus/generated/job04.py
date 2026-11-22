# ruff: noqa
import pyspark.sql.functions as F

# F.nope('x')
df = spark.sql("SELECT tinyint(a) FROM t4")
df = spark.sql("SELECT theta_difference(a) FROM t4")
df = spark.sql("SELECT ascii(a) FROM t4")
