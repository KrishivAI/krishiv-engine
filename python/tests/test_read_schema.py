# SPDX-License-Identifier: Apache-2.0
"""A user schema on ``spark.read`` (``DataFrameReader.schema`` and ``schema=``,
acceptance issues KN-1 and PY-018): the files are read as its columns, never
inferred and never dropped, as a ``StructType`` or a DDL string. Spark 4.2.0
reports every column nullable and puts partition columns last (the goldens in
``tests/fixtures/spark-4.2.0/read-user-schema.json``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, SparkSession
from krishiv.sql.types import IntegerType, LongType, StringType, StructField, StructType

SCHEMA = "LocationID INT, Borough STRING"


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def path(tmp_path: Path) -> str:
    f = tmp_path / "zones.csv"
    f.write_text("LocationID,Borough\n1,EWR\n2,Queens\n10,Bronx\n")
    return str(f)


def shape(df: DataFrame) -> list[tuple[str, str, bool]]:
    return [(f.name, f.dataType.simpleString(), f.nullable) for f in df.schema.fields]


@pytest.mark.parametrize("read", ["csv", "load", "reader"])
def test_a_ddl_schema_types_the_columns(spark: SparkSession, path: str, read: str) -> None:
    calls = {
        "csv": lambda: spark.read.csv(path, header=True, schema=SCHEMA),
        "load": lambda: spark.read.load(path, format="csv", header=True, schema=SCHEMA),
        "reader": lambda: spark.read.schema(SCHEMA).option("header", True).csv(path),
    }
    df = calls[read]()
    assert shape(df) == [("LocationID", "int", True), ("Borough", "string", True)]
    # Numeric, not textual: 10 sorts after 2.
    assert [r[0] for r in df.orderBy("LocationID").collect()] == [1, 2, 10]


def test_a_struct_type_schema_reads_nullable(spark: SparkSession, tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text('{"a":1,"b":"x"}\n{"a":2}\n')
    schema = StructType([StructField("a", LongType(), False), StructField("b", StringType(), False)])
    df = spark.read.schema(schema).json(str(f))
    # Spark reports a read's columns nullable whatever the schema says.
    assert shape(df) == [("a", "bigint", True), ("b", "string", True)]
    assert sorted(tuple(r) for r in df.collect()) == [(1, "x"), (2, None)]


def test_partition_columns_come_last_with_the_schema_type(spark: SparkSession, tmp_path: Path) -> None:
    for k in ("1", "2"):
        d = tmp_path / f"k={k}"
        d.mkdir()
        (d / "f.csv").write_text(f"v{k}\n")
    df = spark.read.schema(StructType([StructField("k", IntegerType()), StructField("v", StringType())])).csv(
        str(tmp_path)
    )
    assert shape(df) == [("v", "string", True), ("k", "int", True)]
    assert sorted(tuple(r) for r in df.collect()) == [("v1", 1), ("v2", 2)]


def test_a_partition_value_that_does_not_cast_fails(spark: SparkSession, tmp_path: Path) -> None:
    d = tmp_path / "k=abc"
    d.mkdir()
    (d / "f.csv").write_text("v\n")
    with pytest.raises(errors.PySparkException) as e:
        spark.read.schema("v STRING, k INT").csv(str(tmp_path)).collect()
    assert e.value.getCondition() == "INVALID_PARTITION_VALUE"


def test_a_schema_of_another_type_is_refused(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkTypeError) as e:
        spark.read.schema(42)  # ty: ignore[invalid-argument-type]
    assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert e.value.getMessageParameters() == {
        "arg_name": "schema",
        "expected_type": "str or struct type",
        "arg_type": "int",
    }


def test_a_bad_ddl_string_is_a_parse_error(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkException) as e:
        spark.read.schema("a INT,,").load("/nowhere", format="csv")
    assert (e.value.getCondition() or "").startswith("PARSE_")


def test_schema_none_reads_as_before(spark: SparkSession, path: str) -> None:
    df = spark.read.csv(path, header=True, inferSchema=True, schema=None)
    assert [f.dataType.simpleString() for f in df.schema.fields] == ["int", "string"]


def test_sql_declares_the_columns(spark: SparkSession, path: str) -> None:
    columns = "LocationID INT, Borough STRING"
    spark.sql(f"CREATE OR REPLACE TEMPORARY VIEW zones ({columns}) USING csv OPTIONS (path '{path}', header 'true')")
    rows = spark.sql("SELECT LocationID FROM zones ORDER BY LocationID").collect()
    assert [r[0] for r in rows] == [1, 2, 10]
