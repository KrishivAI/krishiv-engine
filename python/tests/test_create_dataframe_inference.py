# SPDX-License-Identifier: Apache-2.0
"""``createDataFrame`` over rows without a schema: PySpark's ``_infer_type`` per value and
``_merge_type`` across rows, not Arrow's promotion. The nested types cannot be held by the
engine yet, so their merge rules are checked on the inference helpers and, end to end, by
the errors that come before the engine's own."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import Row, SparkSession
from krishiv.sql import types as T
from krishiv.sql._infer import has_nulltype, infer_row, infer_schema, infer_type, merge_type

pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def test_each_python_type_has_its_spark_type(spark: SparkSession) -> None:
    row = (
        True,
        7,
        1.5,
        "s",
        b"b",
        Decimal("1.5"),
        datetime.datetime(2024, 1, 2, 3, 4, 5),
        datetime.date(2024, 1, 2),
        None,
    )
    df = spark.createDataFrame([row, (False, 8, 2.5, "t", b"c", Decimal("2.5"), row[6], row[7], 1)])
    assert df.dtypes == [
        ("_1", "boolean"),
        ("_2", "bigint"),
        ("_3", "double"),
        ("_4", "string"),
        ("_5", "binary"),
        ("_6", "decimal(38,18)"),
        ("_7", "timestamp"),
        ("_8", "date"),
        ("_9", "bigint"),
    ]
    first = df.orderBy("_2").first()
    assert first is not None
    assert first["_6"] == Decimal("1.5")
    assert first["_9"] is None


def test_void_merges_with_any_type_in_either_order(spark: SparkSession) -> None:
    df = spark.createDataFrame([(None, 1), (2, None), (None, None), (3, 4)], ["a", "b"])
    assert df.dtypes == [("a", "bigint"), ("b", "bigint")]
    assert sorted(df.collect(), key=lambda r: (r.a is None, r.a or 0, r.b is None, r.b or 0)) == [
        Row(a=2, b=None),
        Row(a=3, b=4),
        Row(a=None, b=1),
        Row(a=None, b=None),
    ]
    df = spark.createDataFrame([{"a": 1, "b": None}, {"a": 2, "b": "x"}])
    assert df.dtypes == [("a", "bigint"), ("b", "string")]


def test_equal_types_merge_whatever_the_values(spark: SparkSession) -> None:
    df = spark.createDataFrame([(Decimal("1.5"),), (Decimal("123456789.123456789"),)])
    assert df.dtypes == [("_1", "decimal(38,18)")]
    df = spark.createDataFrame(
        [(datetime.datetime(2024, 1, 1),), (datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),)]
    )
    assert df.dtypes == [("_1", "timestamp")]


@pytest.mark.parametrize(
    ("first", "second", "type1", "type2"),
    [
        (1, 1.5, "LongType", "DoubleType"),
        (1.5, 1, "DoubleType", "LongType"),
        (True, 1, "BooleanType", "LongType"),
        (1, "a", "LongType", "StringType"),
        ("a", b"a", "StringType", "BinaryType"),
        (datetime.date(2024, 1, 1), datetime.datetime(2024, 1, 1), "DateType", "TimestampType"),
        (Decimal("1"), 1, "DecimalType", "LongType"),
    ],
)
def test_other_differences_cannot_merge(spark: SparkSession, first: Any, second: Any, type1: str, type2: str) -> None:
    with pytest.raises(errors.PySparkTypeError) as bad:
        spark.createDataFrame([(first,), (second,)])
    assert bad.value.getCondition() == "CANNOT_MERGE_TYPE"
    assert bad.value.getMessageParameters() == {"data_type1": type1, "data_type2": type2}
    assert str(bad.value) == f"[CANNOT_MERGE_TYPE] Can not merge type `{type1}` and `{type2}`."


def test_the_first_row_that_does_not_merge_is_reported(spark: SparkSession) -> None:
    # Row 2 breaks column b, row 3 would break column a: rows merge in order, as in PySpark.
    rows = [(1, "x"), (2, 1), (1.5, "y")]
    with pytest.raises(errors.PySparkTypeError) as bad:
        spark.createDataFrame(rows, ["a", "b"])
    assert bad.value.getMessageParameters() == {"data_type1": "StringType", "data_type2": "LongType"}


def test_a_column_that_stays_void_cannot_be_determined(spark: SparkSession) -> None:
    for rows in ([(1, None), (2, None)], [(None,)], [{"a": None}], [([],), ([],)], [({},)]):
        with pytest.raises(errors.PySparkValueError) as bad:
            spark.createDataFrame(rows)
        assert bad.value.getCondition() == "CANNOT_DETERMINE_TYPE"
        assert str(bad.value) == "[CANNOT_DETERMINE_TYPE] Some of types cannot be determined after inferring."


def test_nested_values_merge_before_the_engine_rejects_them(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkTypeError) as bad:
        spark.createDataFrame([([1],), ([1.5],)])
    assert bad.value.getCondition() == "CANNOT_MERGE_TYPE"
    assert bad.value.getMessageParameters() == {"data_type1": "LongType", "data_type2": "DoubleType"}
    with pytest.raises(errors.PySparkTypeError) as bad:
        spark.createDataFrame([({"k": 1},), ({"k": "v"},)])
    assert bad.value.getCondition() == "CANNOT_MERGE_TYPE"
    with pytest.raises(errors.PySparkTypeError) as bad:
        spark.createDataFrame([(Row(a=1),), (Row(a="x"),)])
    assert bad.value.getCondition() == "CANNOT_MERGE_TYPE"
    # Merged, the type is the engine's to hold: it has no array type yet.
    with pytest.raises(errors.UnsupportedOperationException) as unsupported:
        spark.createDataFrame([([1, 2],), ([],), ([3],)])
    assert "array<bigint>" in str(unsupported.value)


def test_values_without_a_spark_type_are_unsupported(spark: SparkSession) -> None:
    for value in (datetime.time(1, 2), {1, 2}, object()):
        with pytest.raises(errors.PySparkTypeError) as bad:
            spark.createDataFrame([(value,)])
        assert bad.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
        assert bad.value.getMessageParameters() == {"data_type": type(value).__name__}


def test_infer_type_of_each_kind() -> None:
    assert infer_type(None) == T.NullType()
    assert infer_type(True) == T.BooleanType()
    assert infer_type(2**40) == T.LongType()
    assert infer_type(1.5) == T.DoubleType()
    assert infer_type("s") == T.StringType()
    assert infer_type(b"b") == T.BinaryType()
    assert infer_type(bytearray(b"b")) == T.BinaryType()
    assert infer_type(Decimal("0.1")) == T.DecimalType(38, 18)
    assert infer_type(datetime.date(2024, 1, 1)) == T.DateType()
    assert infer_type(datetime.datetime(2024, 1, 1)) == T.TimestampType()
    # Lists merge their elements, dicts their keys and values (a None skipped).
    assert infer_type([1, None, 2]) == T.ArrayType(T.LongType(), True)
    assert infer_type([]) == T.ArrayType(T.NullType(), True)
    assert infer_type({"a": 1, None: 2, "b": None}) == T.MapType(T.StringType(), T.LongType(), True)
    assert infer_type({}) == T.MapType(T.NullType(), T.NullType(), True)
    assert infer_type([[1], []]) == T.ArrayType(T.ArrayType(T.LongType(), True), True)
    # Tuples and rows are structs named by position or by their fields.
    assert infer_type((1, "a")) == T.StructType(
        [T.StructField("_1", T.LongType(), True), T.StructField("_2", T.StringType(), True)]
    )
    assert infer_type(Row(x=1.5)) == T.StructType([T.StructField("x", T.DoubleType(), True)])
    with pytest.raises(errors.PySparkTypeError, match="UNSUPPORTED_DATA_TYPE"):
        infer_type(object())
    with pytest.raises(errors.PySparkTypeError, match="CANNOT_MERGE_TYPE"):
        infer_type([1, "a"])
    with pytest.raises(errors.PySparkTypeError, match="CANNOT_MERGE_TYPE"):
        infer_type({"a": 1, "b": 1.5})
    with pytest.raises(errors.PySparkTypeError, match="CANNOT_MERGE_TYPE"):
        infer_type({1: 1, "b": 2})


def test_merge_type_rules() -> None:
    long, double, string = T.LongType(), T.DoubleType(), T.StringType()
    void = T.NullType()
    assert merge_type(void, long) == long
    assert merge_type(long, void) == long
    assert merge_type(void, void) == void
    assert merge_type(long, long) == long
    with pytest.raises(errors.PySparkTypeError, match="CANNOT_MERGE_TYPE"):
        merge_type(long, double)
    # Nested types merge recursively and are always nullable.
    assert merge_type(T.ArrayType(void, True), T.ArrayType(long, True)) == T.ArrayType(long, True)
    assert merge_type(T.ArrayType(long, False), T.ArrayType(void, True)) == T.ArrayType(long, True)
    assert merge_type(T.MapType(void, long, False), T.MapType(string, void, True)) == T.MapType(string, long, True)
    with pytest.raises(errors.PySparkTypeError) as bad:
        merge_type(T.ArrayType(long, True), T.ArrayType(double, True))
    assert bad.value.getMessageParameters() == {"data_type1": "LongType", "data_type2": "DoubleType"}
    with pytest.raises(errors.PySparkTypeError) as bad:
        merge_type(T.ArrayType(long, True), T.MapType(long, long, True))
    assert bad.value.getMessageParameters() == {"data_type1": "ArrayType", "data_type2": "MapType"}
    with pytest.raises(errors.PySparkTypeError, match="CANNOT_MERGE_TYPE"):
        merge_type(T.MapType(string, long, True), T.MapType(long, long, True))
    # Structs merge by field name: a field one side lacks is kept, in order of appearance.
    left = T.StructType([T.StructField("a", void), T.StructField("b", long)])
    right = T.StructType([T.StructField("b", void), T.StructField("c", string), T.StructField("a", double)])
    merged = merge_type(left, right)
    assert isinstance(merged, T.StructType)
    assert [(f.name, f.dataType, f.nullable) for f in merged.fields] == [
        ("a", double, True),
        ("b", long, True),
        ("c", string, True),
    ]
    nested = merge_type(
        T.ArrayType(T.StructType([T.StructField("x", void)]), True),
        T.ArrayType(T.StructType([T.StructField("x", long), T.StructField("y", void)]), True),
    )
    assert nested == T.ArrayType(T.StructType([T.StructField("x", long), T.StructField("y", void)]), True)
    with pytest.raises(errors.PySparkTypeError, match="CANNOT_MERGE_TYPE"):
        merge_type(
            T.StructType([T.StructField("x", long)]),
            T.StructType([T.StructField("x", double)]),
        )


def test_has_nulltype_looks_inside_nested_types() -> None:
    void, long = T.NullType(), T.LongType()
    assert has_nulltype(void)
    assert not has_nulltype(long)
    assert has_nulltype(T.ArrayType(void, True))
    assert has_nulltype(T.MapType(long, void, True))
    assert has_nulltype(T.MapType(void, long, True))
    assert has_nulltype(T.StructType([T.StructField("a", long), T.StructField("b", T.ArrayType(void, True))]))
    assert not has_nulltype(T.StructType([T.StructField("a", T.ArrayType(long, True))]))


def test_infer_schema_merges_rows_in_order() -> None:
    schema = infer_schema([[None, "a"], [1, None], [2, "b"]], ["x", "y"])
    assert [(f.name, f.dataType) for f in schema.fields] == [("x", T.LongType()), ("y", T.StringType())]
    assert infer_row(Row(p=1), None).names == ["p"]
    assert infer_row((1, 2), ["u", "v"]).names == ["u", "v"]
    with pytest.raises(errors.PySparkValueError, match="CANNOT_DETERMINE_TYPE"):
        infer_schema([[None, 1], [None, 2]], ["x", "y"])
