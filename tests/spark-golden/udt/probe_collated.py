# SPDX-License-Identifier: Apache-2.0
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
    UserDefinedType,
)


class P:
    def __init__(self, s):
        self.s = s


class LUDT(UserDefinedType):
    @classmethod
    def sqlType(cls):
        return StringType("UTF8_LCASE")

    @classmethod
    def module(cls):
        return "__main__"

    def serialize(self, o):
        return o.s

    def deserialize(self, d):
        return P(d)


class AUDT(UserDefinedType):
    @classmethod
    def sqlType(cls):
        return ArrayType(StringType("UTF8_LCASE"), False)

    @classmethod
    def module(cls):
        return "__main__"

    def serialize(self, o):
        return [o.s]

    def deserialize(self, d):
        return P(d[0])


P.__UDT__ = LUDT()
spark = (
    SparkSession.builder.master("local[1]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
print("VERSION", spark.version)
import re


def show(label, jdt):
    print(
        label,
        re.sub(r'"serializedClass":"[^"]*"', '"serializedClass":"..."', jdt.json()),
    )


for name, u in (("string", LUDT()), ("array", AUDT())):
    df = spark.createDataFrame([], StructType([StructField("p", u)]))
    show("TOPLEVEL-" + name + " ", df._jdf.schema().apply("p").dataType())
    show("STRUCTFIELD-" + name + " ", df._jdf.schema())
    inner = StructType([StructField("s", StructType([StructField("p", u)]))])
    show("NESTED-" + name + " ", spark.createDataFrame([], inner)._jdf.schema())
    with tempfile.TemporaryDirectory() as tmp:
        df.write.mode("overwrite").parquet(tmp + "/udt_coll_" + name)

        rd = spark.read.parquet(tmp + "/udt_coll_" + name)
        print(
            "REREAD-" + name,
            rd.schema.simpleString(),
            rd._jdf.schema().json()[:300].replace("\n", " "),
        )
