# SPDX-License-Identifier: Apache-2.0
"""``StringType(collation)``, checked against PySpark 4.2.0."""

from krishiv.sql.types import DataType, StringType


def test_default_string_is_unchanged() -> None:
    t = StringType()
    assert t.collation == "UTF8_BINARY"
    assert (t.simpleString(), t.jsonValue(), repr(t)) == ("string", "string", "StringType()")
    assert StringType("UTF8_BINARY") == t
    assert hash(StringType("UTF8_BINARY")) == hash(t)


def test_collated_string_prints_and_compares_like_pyspark() -> None:
    t = StringType("utf8_lcase")
    assert t.simpleString() == "string collate utf8_lcase"
    assert t.jsonValue() == "string collate utf8_lcase"
    assert repr(t) == "StringType('utf8_lcase')"
    assert t != StringType()
    # PySpark compares the names as written.
    assert StringType("UTF8_LCASE") != t


def test_ddl_round_trip() -> None:
    assert DataType.fromDDL("string collate UTF8_LCASE") == StringType("UTF8_LCASE")


def test_struct_json_with_collations_metadata() -> None:
    from krishiv.sql.types import ArrayType, MapType, StructType

    field = {
        "type": "struct",
        "fields": [
            {"name": "s", "type": "string", "nullable": True, "metadata": {"__COLLATIONS": {"s": "spark.UTF8_LCASE"}}},
            {
                "name": "a",
                "type": {"type": "array", "elementType": "string", "containsNull": True},
                "nullable": True,
                "metadata": {"__COLLATIONS": {"a.element": "icu.UNICODE_CI"}},
            },
            {
                "name": "m",
                "type": {"type": "map", "keyType": "string", "valueType": "string", "valueContainsNull": True},
                "nullable": True,
                "metadata": {"__COLLATIONS": {"m.key": "spark.UTF8_LCASE"}, "k": 1},
            },
        ],
    }
    t = StructType.fromJson(field)
    assert t["s"].dataType == StringType("UTF8_LCASE")
    assert t["a"].dataType == ArrayType(StringType("UNICODE_CI"))
    assert t["m"].dataType == MapType(StringType("UTF8_LCASE"), StringType())
    assert t["m"].metadata == {"k": 1}
