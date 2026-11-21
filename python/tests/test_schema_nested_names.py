# SPDX-License-Identifier: Apache-2.0
"""``DataFrame.schema`` with nested field names that are not plain identifiers (KN-8)."""

from __future__ import annotations

from krishiv.sql import SparkSession
from krishiv.sql.types import IntegerType, StringType, StructField, StructType


def test_schema_keeps_awkward_nested_names() -> None:
    spark = SparkSession.builder.create()
    df = spark.sql("SELECT named_struct('+1', 1, 'a b', 'x', 'c,d', 2) AS reactions")
    expected = StructType(
        [
            StructField(
                "reactions",
                StructType(
                    [
                        StructField("+1", IntegerType(), False),
                        StructField("a b", StringType(), False),
                        StructField("c,d", IntegerType(), False),
                    ]
                ),
                False,
            )
        ]
    )
    assert df.schema == expected
