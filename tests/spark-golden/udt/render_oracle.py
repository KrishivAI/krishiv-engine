# SPDX-License-Identifier: Apache-2.0
"""Spark 4.2.0 oracle for UDT rendering (P3-T079): spark-submit render_oracle.py in apache/spark:4.2.0."""

import tempfile

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StructField,
    StructType,
    UserDefinedType,
)


class Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


class PtUDT(UserDefinedType):
    @classmethod
    def sqlType(cls):
        return ArrayType(DoubleType(), False)

    @classmethod
    def module(cls):
        return "pt"

    @classmethod
    def scalaUDT(cls):
        return "org.apache.spark.sql.test.ExamplePointUDT"

    def serialize(self, o):
        return [o.x, o.y]

    def deserialize(self, d):
        return Pt(d[0], d[1])


Pt.__UDT__ = PtUDT()
spark = (
    SparkSession.builder.master("local[1]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
schema = StructType([StructField("p", PtUDT())])
df = spark.createDataFrame([(Pt(1.0, 2.0),)], schema)
dt = df.schema["p"].dataType
print("py simple:", dt.simpleString(), "| json:", dt.jsonValue())
jdt = df._jdf.schema().apply("p").dataType()
print(
    "jvm simple:",
    jdt.simpleString(),
    "| catalog:",
    jdt.catalogString(),
    "| sql:",
    jdt.sql(),
    "| typeName:",
    jdt.typeName(),
)
print("jvm json:", jdt.json())
print("schema simple:", df.schema.simpleString())
df.printSchema()
df.createOrReplaceTempView("t")
for r in spark.sql("describe t").collect():
    print("describe:", tuple(r))
print("collect:", df.collect()[0][0].x)
with tempfile.TemporaryDirectory() as tmp:
    df.write.mode("overwrite").parquet(tmp + "/udt_pq")
    print("reread:", spark.read.parquet(tmp + "/udt_pq").schema.simpleString())
