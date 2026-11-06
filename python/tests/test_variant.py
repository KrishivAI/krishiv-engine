# SPDX-License-Identifier: Apache-2.0
"""``VARIANT`` values in Python: ``collect`` returns ``VariantVal`` and ``show`` prints Spark's JSON (Spark 4.2.0:
``parse_json`` / ``to_json`` and PySpark's ``VariantVal.toPython``)."""

from __future__ import annotations

import datetime
import struct
from decimal import Decimal

import pytest

from krishiv.sql import types as T
from krishiv.sql.session import SparkSession


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def test_collect_returns_variant_values(spark: SparkSession) -> None:
    row = spark.sql(
        """SELECT parse_json('{"b":1,"a":[1,2.5,"x",null],"c":{"d":true}}') AS v, parse_json(NULL) AS n,
        CAST(DATE'2024-02-29' AS VARIANT) AS d, CAST(TIMESTAMP_NTZ'2024-01-01 10:00:00.5' AS VARIANT) AS t"""
    ).collect()[0]
    v = row.v
    assert isinstance(v, T.VariantVal)
    assert v.toJson() == '{"a":[1,2.5,"x",null],"b":1,"c":{"d":true}}'
    assert str(v) == v.toJson()
    assert v.toPython() == {"a": [1, Decimal("2.5"), "x", None], "b": 1, "c": {"d": True}}
    assert row.n is None
    assert row.d.toPython() == datetime.date(2024, 2, 29)
    assert row.d.toJson() == '"2024-02-29"'
    assert row.t.toJson() == '"2024-01-01 10:00:00.5"'
    assert row.t.toPython() == datetime.datetime(2024, 1, 1, 10, 0, 0, 500000)
    assert v == T.VariantVal(v.value, v.metadata)


def test_show_prints_json(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    spark.sql("""SELECT parse_json('{"k": "v", "n": 1.50}') AS v""").show()
    out = capsys.readouterr().out
    assert '{"k":"v","n":1.5}' in out


def test_values_print_as_spark() -> None:
    from krishiv.sql.variant import VariantVal

    # A double, a long string and a timestamp, hand-built (primitive ids 7, 16, 12).
    meta = bytes([1, 0, 0])
    double = VariantVal(bytes([7 << 2]) + struct.pack("<d", 1e10), meta)
    assert double.toJson() == "1.0E10"
    text = "z" * 70
    long_string = VariantVal(bytes([16 << 2]) + len(text).to_bytes(4, "little") + text.encode(), meta)
    assert long_string.toPython() == text
    ts = VariantVal(bytes([12 << 2]) + (1_500_000).to_bytes(8, "little", signed=True), meta)
    assert ts.toJson() == '"1970-01-01 00:00:01.5+00:00"'
    assert ts.toJson("+05:30") == '"1970-01-01 05:30:01.5+05:30"'
    assert ts.toPython() == datetime.datetime(1970, 1, 1, 0, 0, 1, 500000, tzinfo=datetime.UTC)
