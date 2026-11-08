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


# Values built from their bytes (empty dictionary). Expected values follow Spark's JVM ``Variant`` (``VariantUtil``,
# SQL ``to_json``), which Krishiv's ``VariantVal`` decodes as; where PySpark 4.2.0's own ``VariantVal`` differs is in
# docs/explanation/known-differences.md ("PySpark's ``VariantVal``").
_META = b"\x01\x00\x00"


def _primitive(type_id: int, body: bytes) -> T.VariantVal:
    return T.VariantVal(bytes([type_id << 2]) + body, _META)


def test_primitive_variants_decode_as_spark() -> None:
    ntz = _primitive(13, struct.pack("<q", 1_700_000_000_123_456))
    assert ntz.toJson() == '"2023-11-14 22:13:20.123456"'
    assert ntz.toPython() == datetime.datetime(2023, 11, 14, 22, 13, 20, 123456)
    binary = _primitive(15, struct.pack("<I", 3) + b"\x00\xffa")
    assert binary.toJson() == '"AP9h"'
    assert binary.toPython() == b"\x00\xffa"
    text = _primitive(16, struct.pack("<I", 5) + b"hello")
    assert text.toJson() == '"hello"'
    assert text.toPython() == "hello"
    assert _primitive(7, struct.pack("<d", 1.5)).toPython() == 1.5
    assert _primitive(7, struct.pack("<d", 1.5)).toJson() == "1.5"
    assert _primitive(14, struct.pack("<f", 2.5)).toPython() == 2.5
    assert _primitive(14, struct.pack("<f", 2.5)).toJson() == "2.5"


def test_string_escapes_follow_spark() -> None:
    raw = b'a"b\\c\n\x01'
    v = T.VariantVal(bytes([(len(raw) << 2) | 1]) + raw, _META)
    assert v.toJson() == '"a\\"b\\\\c\\n\\u0001"'
    assert v.toPython() == 'a"b\\c\n\x01'


def test_uuid_and_identity() -> None:
    import uuid

    u = uuid.UUID("12345678-1234-5678-1234-567812345678")
    v = _primitive(20, u.bytes)
    assert v.toPython() == u
    assert v.toJson() == '"12345678-1234-5678-1234-567812345678"'
    same = _primitive(20, u.bytes)
    assert v == same
    assert hash(v) == hash(same)
    assert repr(v) == f"VariantVal({v.value!r}, {v.metadata!r})"
    assert str(v) == v.toJson()


def test_bad_metadata_and_unknown_types_raise() -> None:
    with pytest.raises(ValueError, match="MALFORMED_VARIANT"):
        T.VariantVal(_primitive(1, b"").value, b"\x02\x00\x00").toJson()
    with pytest.raises(ValueError, match="UNKNOWN_PRIMITIVE_TYPE_IN_VARIANT"):
        _primitive(21, b"").toJson()
    with pytest.raises(ValueError, match="UNKNOWN_PRIMITIVE_TYPE_IN_VARIANT"):
        _primitive(21, b"").toPython()
