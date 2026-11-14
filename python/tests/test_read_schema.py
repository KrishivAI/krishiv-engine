# SPDX-License-Identifier: Apache-2.0
"""A user schema on ``spark.read`` is refused, never dropped (acceptance issue
KN-1): a dropped ``schema=`` read every column as a string without a word, so
comparisons and sorts were textual. Reads do not carry a user schema yet; SQL
declares the columns instead."""

from __future__ import annotations

from pathlib import Path

import pytest

from krishiv import errors
from krishiv.sql import SparkSession

SCHEMA = "LocationID INT, Borough STRING"


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def path(tmp_path: Path) -> str:
    f = tmp_path / "zones.csv"
    f.write_text("LocationID,Borough\n1,EWR\n2,Queens\n10,Bronx\n")
    return str(f)


def refused(e: pytest.ExceptionInfo[errors.UnsupportedOperationException]) -> None:
    assert e.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "user schema on spark.read" in str(e.value)


@pytest.mark.parametrize("read", ["csv", "json", "load"])
def test_a_schema_argument_is_refused(spark: SparkSession, path: str, read: str) -> None:
    calls = {
        "csv": lambda: spark.read.csv(path, header=True, schema=SCHEMA),
        "json": lambda: spark.read.json(path, schema=SCHEMA),
        "load": lambda: spark.read.load(path, format="csv", header=True, schema=SCHEMA),
    }
    with pytest.raises(errors.UnsupportedOperationException) as e:
        calls[read]()
    refused(e)


def test_reader_schema_is_refused(spark: SparkSession) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as e:
        spark.read.schema(SCHEMA)
    refused(e)


def test_schema_none_reads_as_before(spark: SparkSession, path: str) -> None:
    df = spark.read.csv(path, header=True, inferSchema=True, schema=None)
    assert [f.dataType.simpleString() for f in df.schema.fields] == ["int", "string"]


def test_sql_declares_the_columns(spark: SparkSession, path: str) -> None:
    columns = "LocationID INT, Borough STRING"
    spark.sql(f"CREATE OR REPLACE TEMPORARY VIEW zones ({columns}) USING csv OPTIONS (path '{path}', header 'true')")
    rows = spark.sql("SELECT LocationID FROM zones ORDER BY LocationID").collect()
    assert [r[0] for r in rows] == [1, 2, 10]
