# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: BLE001, UP031  (a probe script: it prints what Spark does, including its errors)
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


class PyOnlyUDT(UserDefinedType):
    @classmethod
    def sqlType(cls):
        return ArrayType(DoubleType(), False)

    @classmethod
    def module(cls):
        return "__main__"

    def serialize(self, o):
        return [o.x, o.y]

    def deserialize(self, d):
        return Pt(d[0], d[1])


Pt.__UDT__ = PyOnlyUDT()
spark = (
    SparkSession.builder.master("local[1]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
print("VERSION", spark.version)
df = spark.createDataFrame(
    [(Pt(1.0, 2.0),)], StructType([StructField("p", PyOnlyUDT())])
)
j = df._jdf.schema().apply("p").dataType()
print("PYONLY jvm class:", j.getClass().getName())
print("PYONLY jvm json:", j.json())
print(
    "PYONLY simple/typeName/catalog/sql:",
    j.simpleString(),
    j.typeName(),
    j.catalogString(),
    j.sql(),
)
df.printSchema()
with tempfile.TemporaryDirectory() as tmp:
    df.write.mode("overwrite").parquet(tmp + "/udt_pq_pyonly")
    print(
        "PARQUET reread:",
        spark.read.parquet(tmp + "/udt_pq_pyonly").schema.simpleString(),
    )
DT = spark._jvm.org.apache.spark.sql.types.DataType
sql = '{"type":"array","elementType":"double","containsNull":false}'
for name, js in [
    (
        "jvm-null-pyclass",
        '{"type":"udt","class":"org.apache.spark.sql.test.ExamplePointUDT","pyClass":null,"sqlType":%s}'
        % sql,
    ),
    (
        "jvm-no-pyclass",
        '{"type":"udt","class":"org.apache.spark.sql.test.ExamplePointUDT","sqlType":%s}'
        % sql,
    ),
    (
        "py-only",
        '{"type":"udt","pyClass":"pt.PyOnlyUDT","serializedClass":"gASV","sqlType":%s}'
        % sql,
    ),
    (
        "py-only-no-serialized",
        '{"type":"udt","pyClass":"pt.PyOnlyUDT","sqlType":%s}' % sql,
    ),
]:
    try:
        r = DT.fromJson(js)
        print("PARSE", name, "OK", r.getClass().getName(), r.simpleString())
    except Exception as e:
        print("PARSE", name, "ERR", str(e).splitlines()[0][:200])
