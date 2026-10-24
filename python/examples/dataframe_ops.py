# SPDX-License-Identifier: Apache-2.0
"""Typed example for ``mypy --strict``: transformations, types and rows."""

from krishiv.sql import DataFrame, SparkSession
from krishiv.sql import functions as F
from krishiv.sql import types as T
from krishiv.sql.row import Row


def build(spark: SparkSession) -> DataFrame:
    df: DataFrame = spark.sql("SELECT 1 AS id, 'a' AS name")
    grown: DataFrame = df.withColumn("v", F.col("id") + F.lit(1))
    return grown.filter(F.col("id") > F.lit(0))


def describe(df: DataFrame) -> str:
    return f"{df.columns}: {df.schema.simpleString()}"


def first_id(df: DataFrame) -> int | None:
    rows: list[Row] = df.take(1)
    if not rows:
        return None
    value: object = rows[0]["id"]
    assert isinstance(value, int)
    return value


def main() -> None:
    spark: SparkSession = SparkSession.builder.create()
    df: DataFrame = build(spark)
    print(describe(df))
    print(first_id(df))
    field: T.StructField = df.schema["id"]
    print(field.dataType.simpleString())


if __name__ == "__main__":
    main()
