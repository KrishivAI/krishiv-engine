# SPDX-License-Identifier: Apache-2.0
# Builds views holding variants whose bytes cannot be produced by SQL (executed by ../../run_oracle.py).
from pyspark.sql.types import (
    IntegerType,
    StructField,
    StructType,
    VariantType,
    VariantVal,
)

# metadata: version 1, dict size 0 ; value: header byte, see Spark variant spec
good_meta = bytes([0x01, 0x00, 0x00])
rows = [
    (1, VariantVal(bytes([0x0C, 0x01]), good_meta)),  # int8 1  -> valid
    (2, VariantVal(bytes([0x0C]), good_meta)),  # truncated int8
    (3, VariantVal(bytes([0xFF]), good_meta)),  # unknown basic type/ids
    (
        4,
        VariantVal(bytes([0x0C, 0x01]), bytes([0x02, 0x00, 0x00])),
    ),  # bad metadata version
    (5, VariantVal(bytes([]), good_meta)),  # empty value
    (6, VariantVal(bytes([0x0C, 0x01]), bytes([]))),  # empty metadata
    (
        7,
        VariantVal(bytes([0x02, 0x01, 0x00, 0x00, 0x05]), good_meta),
    ),  # object, field id out of range
    (8, VariantVal(bytes([0x0C, 0x01, 0x99]), good_meta)),  # trailing garbage
    (9, None),
]
schema = StructType([StructField("id", IntegerType()), StructField("v", VariantType())])
spark.createDataFrame(rows, schema).createOrReplaceTempView("mv")  # noqa: F821
