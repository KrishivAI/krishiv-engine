# SPDX-License-Identifier: Apache-2.0
"""``spark.sql(query, args, **kwargs)``: named and positional parameters (QA2-26)."""

from __future__ import annotations

import datetime
import decimal

import pytest

from krishiv import errors
from krishiv.sql import SparkSession


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def test_named_args_dict_and_kwargs(spark: SparkSession) -> None:
    assert spark.sql("SELECT :a + 1 AS x, :b AS y", args={"a": 1, "b": "it's"}).collect()[0].asDict() == {
        "x": 2,
        "y": "it's",
    }
    assert spark.sql("SELECT :a + :b AS s", a=1, b=2).collect()[0]["s"] == 3


def test_positional_args(spark: SparkSession) -> None:
    row = spark.sql("SELECT ? AS a, ? AS b", args=[1.5, None]).collect()[0]
    assert (row["a"], row["b"]) == (1.5, None)
    assert spark.sql("SELECT ? AS a", args=("x",)).collect()[0]["a"] == "x"


def test_value_types(spark: SparkSession) -> None:
    row = spark.sql(
        "SELECT :d AS d, :t AS t, :c AS c, :n AS n, :bb AS bb, :f AS f, :big AS big",
        d=datetime.date(2020, 1, 2),
        t=datetime.datetime(2020, 1, 2, 3, 4, 5),
        c=decimal.Decimal("1.50"),
        n=float("nan"),
        bb=b"\x01\xff",
        f=True,
        big=2**62,
    ).collect()[0]
    assert row["d"] == datetime.date(2020, 1, 2)
    assert row["t"] == datetime.datetime(2020, 1, 2, 3, 4, 5)
    assert row["c"] == decimal.Decimal("1.50")
    assert row["n"] != row["n"]
    assert row["bb"] == bytearray(b"\x01\xff")
    assert row["f"] is True
    assert row["big"] == 2**62


def test_strings_with_quotes_and_backslashes(spark: SparkSession) -> None:
    assert spark.sql("SELECT :s AS s", s="a'b\\c").collect()[0]["s"] == "a'b\\c"
    assert spark.sql("SELECT ':s' AS s, :t AS t", t="q'").collect()[0]["s"] == ":s"


def test_bad_arguments(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkValueError):
        spark.sql("SELECT :a", args={"a": 1}, a=2)
    with pytest.raises(errors.PySparkTypeError):
        spark.sql("SELECT ?", args=5)
    with pytest.raises(errors.PySparkTypeError):
        spark.sql("SELECT :a", a=object())
    with pytest.raises(errors.PySparkValueError):
        spark.sql("SELECT :a", a=2**70)
    with pytest.raises(errors.AnalysisException) as e:
        spark.sql("SELECT :a, :b", a=1).collect()
    assert e.value.getCondition() == "UNBOUND_SQL_PARAMETER"
