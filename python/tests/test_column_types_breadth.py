# SPDX-License-Identifier: Apache-2.0
"""P2-T186 part B: Column methods, type classes, GroupedData, createDataFrame, range and catalog.

Every test follows the PySpark 4.2 docstring example for its method; where the
engine has no equivalent yet, the test asserts the engine's own error (the same
the SQL equivalent fails with) instead of emulating the feature.
"""

from __future__ import annotations

import datetime
import math
from decimal import Decimal
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import functions as F
from krishiv.sql import types as T
from krishiv.sql.dataframe import DataFrame
from krishiv.sql.row import Row
from krishiv.sql.session import SparkSession

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])


# Column.when / Column.otherwise -------------------------------------------------


def test_when_otherwise(df: DataFrame) -> None:
    out = df.select(df.name, F.when(df.age > 3, 1).otherwise(0))
    assert out.columns == ["name", "CASE WHEN (age > 3) THEN 1 ELSE 0 END"]
    assert out.collect() == [Row("Alice", 0), Row("Bob", 1)]
    assert df.select(F.when(df.age > 4, 1).when(df.age < 3, -1).otherwise(0)).collect() == [
        Row(-1),
        Row(1),
    ]


def test_when_without_otherwise_is_null(df: DataFrame) -> None:
    assert df.select(F.when(df.age > 3, 1).alias("w")).collect() == [Row(w=None), Row(w=1)]


def test_when_needs_an_open_when_chain(df: DataFrame) -> None:
    # As in PySpark 4.2: `when` only extends a `when` chain without an ELSE.
    for bad in (lambda: df.age.when(df.age > 3, 1), lambda: F.when(df.age > 3, 1).otherwise(0).when(df.age > 1, 2)):
        with pytest.raises(errors.PySparkTypeError, match=r"\[INVALID_WHEN_USAGE\] when\(\) can only be applied"):
            bad()


def test_when_takes_columns_and_literals(df: DataFrame) -> None:
    assert df.select(F.when(df.age > 3, df.age).otherwise(0).alias("w")).collect() == [
        Row(w=0),
        Row(w=5),
    ]
    assert df.select(F.when(df.age > 3, F.lit("y")).otherwise(F.lit("n")).alias("w")).collect() == [
        Row(w="n"),
        Row(w="y"),
    ]
    # A scanned string column as a branch fails exactly as the SQL CASE does
    # (a pre-existing engine gap with string storage in CASE).
    spark = df.sparkSession
    df.createOrReplaceTempView("when_people")
    try:
        with pytest.raises(errors.SparkRuntimeException) as py:
            df.select(F.when(df.age > 3, df.name).otherwise(df.name)).collect()
        with pytest.raises(errors.SparkRuntimeException) as sql:
            spark.sql("SELECT CASE WHEN age > 3 THEN name ELSE name END AS w FROM when_people").collect()
        assert type(py.value) is type(sql.value)
        assert str(py.value) == str(sql.value)
    finally:
        spark.catalog.dropTempView("when_people")
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        F.when("age > 3", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        df.age.when("age > 3", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        F.when(df.age > 3, object())


def test_otherwise_needs_when(df: DataFrame) -> None:
    # As in PySpark 4.2: `otherwise` needs a `when` chain, and applies once.
    for bad in (lambda: df.age.otherwise(0), lambda: F.when(df.age > 3, 1).otherwise(0).otherwise(2)):
        with pytest.raises(errors.PySparkTypeError, match="can only be applied once on a Column previously generated"):
            bad()


# contains / startswith / endswith / like / rlike / ilike / substr -------------


def test_contains_startswith_endswith(df: DataFrame) -> None:
    assert df.filter(df.name.contains("o")).collect() == [Row(5, "Bob")]
    assert df.filter(df.name.startswith("Al")).collect() == [Row(2, "Alice")]
    assert df.filter(df.name.startswith("^Al")).collect() == []
    assert df.filter(df.name.endswith("ice")).collect() == [Row(2, "Alice")]
    assert df.filter(df.name.endswith("ice$")).collect() == []
    assert df.filter(df.name.contains(F.lit("o"))).collect() == [Row(5, "Bob")]
    with pytest.raises(errors.PySparkTypeError):
        df.name.contains(object())


def test_like(df: DataFrame) -> None:
    assert df.filter(df.name.like("Al%")).collect() == [Row(2, "Alice")]
    with pytest.raises(errors.PySparkTypeError):
        df.name.like(1)  # ty: ignore[invalid-argument-type]


def test_rlike_ilike(df: DataFrame) -> None:
    # PySpark's docstring examples.
    assert df.filter(df.name.rlike("ice$")).collect() == [Row(age=2, name="Alice")]
    assert df.filter(df.name.ilike("%ICE")).collect() == [Row(age=2, name="Alice")]
    with pytest.raises(errors.PySparkTypeError):
        df.name.rlike(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.name.ilike(None)  # ty: ignore[invalid-argument-type]


def test_substr(spark: SparkSession, df: DataFrame) -> None:
    assert df.select(df.name.substr(1, 3).alias("col")).collect() == [Row(col="Ali"), Row(col="Bob")]
    indexed = spark.createDataFrame([(3, 4, "Alice"), (2, 3, "Bob")], ["sidx", "eidx", "name"])
    assert indexed.select(indexed.name.substr(indexed.sidx, indexed.eidx).alias("col")).collect() == [
        Row(col="ice"),
        Row(col="ob"),
    ]
    assert df.select(df.name[1:3].alias("s")).collect() == [Row(s="Ali"), Row(s="Bob")]
    with pytest.raises(errors.PySparkTypeError, match="NOT_SAME_TYPE"):
        df.name.substr(1, F.lit(2))
    with pytest.raises(errors.PySparkTypeError, match="NOT_SAME_TYPE"):
        df.name.substr("1", 2)
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        df.name.substr("1", "2")
    with pytest.raises(errors.PySparkValueError, match="SLICE_WITH_STEP"):
        df.name[1:3:2]


# bitwise -----------------------------------------------------------------------


def test_bitwise_matches_sql(spark: SparkSession) -> None:
    # PySpark's docstring values, and the same as the SQL operators.
    df = spark.createDataFrame([(170, 75)], ["a", "b"])
    for method, op, want in (("bitwiseOR", "|", 235), ("bitwiseAND", "&", 10), ("bitwiseXOR", "^", 225)):
        got = df.select(getattr(df.a, method)(df.b).alias("x")).collect()
        sql = spark.sql(f"SELECT a {op} b AS x FROM (SELECT 170 AS a, 75 AS b)").collect()
        assert got == sql == [Row(x=want)], method


# isNaN -------------------------------------------------------------------------


def test_isnan_matches_sql(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1.0,), (float("nan"),)], ["x"])
    [row] = df.filter(df.x.isNaN()).collect()
    assert math.isnan(row.x)
    df.createOrReplaceTempView("isnan_input")
    try:
        assert [r.n for r in spark.sql("SELECT isnan(x) AS n FROM isnan_input").collect()] == [False, True]
        assert [r.n for r in df.select(df.x.isNaN().alias("n")).collect()] == [False, True]
    finally:
        spark.catalog.dropTempView("isnan_input")


# getItem / getField / withField / dropFields ------------------------------------


def test_nested_access_is_unsupported(df: DataFrame) -> None:
    for build in (
        lambda: F.col("l").getItem(0),
        lambda: F.col("r").getField("b"),
        lambda: F.col("a").withField("b", F.lit(3)),
        lambda: F.col("a").dropFields("b"),
        lambda: F.col("a").dropFields("b", "c"),
        lambda: df.name.foo,
        lambda: df.name["key"],
    ):
        with pytest.raises(errors.UnsupportedOperationException) as bad:
            build()
        assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    with pytest.warns(FutureWarning), pytest.raises(errors.UnsupportedOperationException):
        F.col("l").getItem(F.col("k"))
    with pytest.warns(FutureWarning), pytest.raises(errors.UnsupportedOperationException):
        F.col("r").getField(F.col("n"))
    with pytest.raises(errors.PySparkTypeError):
        F.col("a").withField(1, F.lit(1))  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        F.col("a").withField("b", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        F.col("a").dropFields(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(AttributeError):
        _ = df.name.__dunder__


# try_cast ----------------------------------------------------------------------


def test_try_cast(spark: SparkSession) -> None:
    df = spark.createDataFrame([("123",), ("Bob",), (None,)], ["name"])
    assert df.select(df.name.try_cast("int").alias("n")).collect() == [
        Row(n=123),
        Row(n=None),
        Row(n=None),
    ]
    assert df.select(df.name.try_cast(T.LongType()).alias("n")).dtypes == [("n", "bigint")]
    with pytest.raises(errors.PySparkTypeError):
        df.name.try_cast(1)  # ty: ignore[invalid-argument-type]
    # Arrays are a supported type; a string simply cannot be cast to one.
    with pytest.raises(errors.AnalysisException) as bad:
        _ = df.select(df.name.try_cast(T.ArrayType(T.StringType()))).columns
    assert bad.value.getCondition() == "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION"


# types ---------------------------------------------------------------------------


def test_atomic_types() -> None:
    assert T.NullType.typeName() == T.NullType().typeName() == "void"
    assert T.IntegerType.typeName() == "integer"
    assert T.TimestampNTZType.typeName() == "timestamp_ntz"
    assert T.CalendarIntervalType.typeName() == "interval"
    assert T.IntegerType() == T.IntegerType()
    assert T.IntegerType() != T.LongType()
    assert hash(T.DecimalType(5, 2)) == hash(T.DecimalType(5, 2))
    assert repr(T.StringType()) == "StringType()"
    assert repr(T.DecimalType(5, 2)) == "DecimalType(5,2)"
    assert T.BooleanType().json() == '"boolean"'
    assert T.DecimalType(10, 2).json() == '"decimal(10,2)"'
    for t in (
        T.NullType,
        T.BooleanType,
        T.ByteType,
        T.ShortType,
        T.FloatType,
        T.BinaryType,
        T.TimestampType,
        T.CalendarIntervalType,
    ):
        assert T._from_simple_string(t().simpleString()) == t()
    with pytest.raises(errors.PySparkValueError):
        T._from_simple_string("array<int>")


def test_char_varchar() -> None:
    assert T.CharType(3).simpleString() == "char(3)"
    assert T.VarcharType(10).simpleString() == "varchar(10)"
    assert repr(T.CharType(3)) == "CharType(3)"
    assert repr(T.VarcharType(10)) == "VarcharType(10)"
    assert T.CharType(3).json() == '"char(3)"'
    assert T._parse_datatype_json_string('"varchar(10)"') == T.VarcharType(10)
    assert T.DataType.fromDDL("char(3)") == T.CharType(3)
    assert T.DataType.fromDDL("VARCHAR(10)") == T.VarcharType(10)
    assert T.DataType.fromDDL("char(3)").json() == '"char(3)"'


def test_interval_types() -> None:
    assert T.DayTimeIntervalType().simpleString() == "interval day to second"
    assert T.YearMonthIntervalType().simpleString() == "interval year to month"
    assert T.DayTimeIntervalType(1).simpleString() == "interval hour"
    assert T.DayTimeIntervalType(1, 2).simpleString() == "interval hour to minute"
    assert T.YearMonthIntervalType(1, 1).simpleString() == "interval month"
    assert repr(T.DayTimeIntervalType()) == "DayTimeIntervalType(0, 3)"
    assert T.DayTimeIntervalType().json() == '"interval day to second"'
    for text, type_ in (
        ("interval day to second", T.DayTimeIntervalType()),
        ("interval hour", T.DayTimeIntervalType(1)),
        ("interval year to month", T.YearMonthIntervalType()),
        ("interval month", T.YearMonthIntervalType(1, 1)),
    ):
        assert T.DataType.fromDDL(text) == type_
        assert T._parse_datatype_json_string(f'"{text}"') == type_
    assert T.DataType.fromDDL("interval") == T.CalendarIntervalType()
    with pytest.raises(errors.PySparkRuntimeError):
        T.DayTimeIntervalType(7)
    with pytest.raises(errors.PySparkRuntimeError):
        T.YearMonthIntervalType(0, 3)
    with pytest.raises(errors.PySparkTypeError):
        T.ArrayType("int")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        T.MapType(T.StringType(), "int")  # ty: ignore[invalid-argument-type]


def test_array_map_types() -> None:
    assert T.ArrayType(T.StringType()) == T.ArrayType(T.StringType(), True)
    assert T.ArrayType(T.StringType(), False) != T.ArrayType(T.StringType())
    assert T.ArrayType(T.StringType()).simpleString() == "array<string>"
    assert T.MapType(T.StringType(), T.IntegerType()) == T.MapType(T.StringType(), T.IntegerType(), True)
    assert T.MapType(T.StringType(), T.IntegerType(), False) != T.MapType(T.StringType(), T.FloatType())
    assert T.MapType(T.StringType(), T.IntegerType()).simpleString() == "map<string,int>"
    nested = T.ArrayType(T.MapType(T.StringType(), T.ArrayType(T.IntegerType())))
    assert nested.simpleString() == "array<map<string,array<int>>>"
    assert T._parse_datatype_json_string(nested.json()) == nested
    assert T.DataType.fromDDL(nested.simpleString()) == nested
    assert T.ArrayType.fromJson({"type": "array", "elementType": "string", "containsNull": False}) == (
        T.ArrayType(T.StringType(), False)
    )
    assert T.MapType.fromJson(
        {
            "type": "map",
            "keyType": "string",
            "valueType": "integer",
            "valueContainsNull": False,
        }
    ) == T.MapType(T.StringType(), T.IntegerType(), False)


def test_struct_add() -> None:
    struct1 = T.StructType().add("f1", T.StringType(), True).add("f2", T.StringType(), True, None)
    struct2 = T.StructType([T.StructField("f1", T.StringType(), True), T.StructField("f2", T.StringType(), True, None)])
    assert struct1 == struct2
    assert T.StructType().add(T.StructField("f1", T.StringType(), True)) == T.StructType(
        [T.StructField("f1", T.StringType(), True)]
    )
    assert T.StructType().add("f1", "string", True) == T.StructType([T.StructField("f1", T.StringType(), True)])
    assert T.StructType().add("f1", "decimal(10,2)", True) == T.StructType(
        [T.StructField("f1", T.DecimalType(10, 2), True)]
    )
    with pytest.raises(errors.PySparkValueError, match="ARGUMENT_REQUIRED"):
        T.StructType().add("f1")
    with pytest.raises(errors.PySparkTypeError):
        T.StructType().add("f1", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(AssertionError):
        T.StructType([T.StructField("f1", T.StringType(), True), "nope"])  # ty: ignore[invalid-argument-type]


def test_struct_json_round_trip() -> None:
    schema = T.StructType(
        [
            T.StructField("name", T.StringType(), True),
            T.StructField(
                "languages",
                T.ArrayType(T.StringType()),
                True,
                {"name": "age"},
            ),
        ]
    )
    assert T.StructType.fromJson(schema.jsonValue()) == schema
    assert T._parse_datatype_json_string(schema.json()) == schema
    field = T.StructField("a", T.LongType())
    assert T.StructField.fromJson(field.jsonValue()) == field
    with pytest.raises(errors.PySparkValueError, match="UNSUPPORTED_DATA_TYPE"):
        T._parse_datatype_json_value({"type": "nope"})
    with pytest.raises(errors.PySparkValueError, match="CANNOT_PARSE_DATATYPE"):
        T._parse_datatype_json_value("nope")


def test_from_ddl() -> None:
    assert T.DataType.fromDDL("int") == T.IntegerType()
    assert T.DataType.fromDDL("INT") == T.IntegerType()
    assert T.DataType.fromDDL("byte") == T.ByteType()
    assert T.DataType.fromDDL("decimal") == T.DecimalType(10, 0)
    assert T.DataType.fromDDL("decimal(16, 8)") == T.DecimalType(16, 8)
    assert T.DataType.fromDDL("a: byte, b: decimal(  16 , 8   ) ") == T.StructType(
        [T.StructField("a", T.ByteType(), True), T.StructField("b", T.DecimalType(16, 8), True)]
    )
    assert T.DataType.fromDDL("a DOUBLE, b STRING") == T.StructType(
        [T.StructField("a", T.DoubleType(), True), T.StructField("b", T.StringType(), True)]
    )
    assert T.DataType.fromDDL("a: array< short>") == T.StructType(
        [T.StructField("a", T.ArrayType(T.ShortType()), True)]
    )
    assert T.DataType.fromDDL(" map<string , string > ") == T.MapType(T.StringType(), T.StringType())
    assert T.DataType.fromDDL("`a.b` int, c string") == T.StructType(
        [T.StructField("a.b", T.IntegerType(), True), T.StructField("c", T.StringType(), True)]
    )
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("blabla")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("a: int,")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("array<int")


def test_struct_type_helpers() -> None:
    schema = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
    assert schema["b"].dataType == T.StringType()
    assert schema[0].name == "a"
    assert schema[1:].fields == [T.StructField("b", T.StringType())]
    assert schema.fieldNames() == ["a", "b"]
    assert len(schema) == 2
    with pytest.raises(errors.PySparkKeyError):
        schema["z"]
    with pytest.raises(errors.PySparkIndexError):
        schema[5]
    with pytest.raises(errors.PySparkTypeError):
        schema[1.5]  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        T.StructField("a", "int")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        T.StructField("x", T.IntegerType()).typeName()


# GroupedData -------------------------------------------------------------------


def test_group_shortcuts(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [(2, "Alice", 80), (3, "Alice", 100), (5, "Bob", 120), (10, "Bob", 140)],
        ["age", "name", "height"],
    )
    assert df.groupBy("name").sum("age").sort("name").collect() == [Row("Alice", 5), Row("Bob", 15)]
    assert df.groupBy("name").avg("age").sort("name").collect() == [Row("Alice", 2.5), Row("Bob", 7.5)]
    assert df.groupBy("name").mean("age").sort("name").collect() == [Row("Alice", 2.5), Row("Bob", 7.5)]
    assert df.groupBy("name").min("age").sort("name").collect() == [Row("Alice", 2), Row("Bob", 5)]
    assert df.groupBy("name").max("age").sort("name").collect() == [Row("Alice", 3), Row("Bob", 10)]
    assert df.groupBy().sum("age", "height").collect() == [Row(20, 440)]
    assert df.groupBy("name").count().sort("name").collect() == [Row("Alice", 2), Row("Bob", 2)]


def test_group_shortcut_no_columns(spark: SparkSession) -> None:
    df = spark.createDataFrame([(2, "Alice", 80), (3, "Alice", 100)], ["age", "name", "height"])
    assert df.groupBy("name").sum().sort("name").collect() == [Row("Alice", 5, 180)]
    assert df.groupBy("name").avg().sort("name").collect() == [Row("Alice", 2.5, 90.0)]


def test_group_shortcut_numeric_columns(spark: SparkSession) -> None:
    df = spark.createDataFrame([(2, "Alice"), (3, "Alice")], ["age", "name"])
    # No columns: the numeric ones. A non-numeric one named fails, as in PySpark 4.2.
    assert df.groupBy("name").sum().sort("name").collect() == [Row("Alice", 5)]
    with pytest.raises(errors.PySparkTypeError, match=r"\[NOT_NUMERIC_COLUMNS\] .* got \['name'\]"):
        df.groupBy("name").sum("age", "name")
    with pytest.raises(errors.AnalysisException):
        _ = df.groupBy("name").sum("missing").columns
    with pytest.raises(errors.PySparkTypeError):
        df.groupBy("name").sum(1)  # ty: ignore[invalid-argument-type]


def test_agg_dict(spark: SparkSession) -> None:
    df = spark.createDataFrame([(2, "Alice"), (3, "Alice"), (5, "Bob")], ["age", "name"])
    assert df.groupBy("name").agg({"*": "count"}).sort("name").collect() == [
        Row("Alice", 2),
        Row("Bob", 1),
    ]
    assert df.groupBy("name").agg({"age": "max"}).sort("name").collect() == [
        Row("Alice", 3),
        Row("Bob", 5),
    ]
    assert df.agg({"age": "min"}).collect() == [Row(2)]  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.groupBy("name").agg({1: "count"})  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.groupBy("name").agg({"age": 1})  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="should not be empty"):
        df.groupBy("name").agg()
    with pytest.raises(errors.PySparkTypeError):
        df.groupBy("name").agg("age")  # ty: ignore[invalid-argument-type]


def test_pivot_is_unsupported(df: DataFrame) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.groupBy("name").pivot("name")
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    with pytest.raises(errors.UnsupportedOperationException):
        df.groupBy("name").pivot("name", ["Alice"])
    with pytest.raises(errors.PySparkTypeError):
        df.groupBy("name").pivot(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.groupBy("name").pivot("name", "Alice")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.groupBy("name").pivot("name", [1, object()])


# spark.range ---------------------------------------------------------------------


def test_range(spark: SparkSession) -> None:
    assert spark.range(1, 7, 2).collect() == [Row(1), Row(3), Row(5)]
    assert spark.range(3).collect() == [Row(0), Row(1), Row(2)]
    assert spark.range(0, 3).collect() == [Row(0), Row(1), Row(2)]
    assert spark.range(5, 0, -2).collect() == [Row(5), Row(3), Row(1)]
    assert spark.range(3, 3).collect() == []
    assert spark.range(1, 7, 2, 4).collect() == [Row(1), Row(3), Row(5)]
    assert spark.range(2).dtypes == [("id", "bigint")]
    with pytest.raises(ValueError, match="step cannot be 0"):
        spark.range(1, 5, 0)
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        spark.range("a")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        spark.range(1, 5, 1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        spark.range(3, numPartitions="x")  # ty: ignore[invalid-argument-type]


# createDataFrame -----------------------------------------------------------------


def test_create_rows_with_schema(spark: SparkSession) -> None:
    assert spark.createDataFrame([("Alice", 1)], ["name", "age"]).collect() == [Row("Alice", 1)]
    schema = T.StructType([T.StructField("name", T.StringType(), True), T.StructField("age", T.IntegerType(), True)])
    assert spark.createDataFrame([("Alice", 1)], schema).collect() == [Row("Alice", 1)]
    assert spark.createDataFrame([("Alice", 1)], "name: string, age: int").collect() == [Row("Alice", 1)]
    assert spark.createDataFrame([("Alice", 1)], "name string, age int").collect() == [Row("Alice", 1)]
    assert spark.createDataFrame([], schema).collect() == []


def test_create_dicts_and_rows(spark: SparkSession) -> None:
    schema = T.StructType([T.StructField("name", T.StringType(), True), T.StructField("age", T.IntegerType(), True)])
    assert spark.createDataFrame([{"name": "Alice", "age": 1}, {"name": "Bob"}], schema).collect() == [
        Row("Alice", 1),
        Row("Bob", None),
    ]
    assert spark.createDataFrame([Row(name="Alice", age=1), Row("Bob", 2)], schema).collect() == [
        Row("Alice", 1),
        Row("Bob", 2),
    ]
    assert spark.createDataFrame([Row("Alice", 1)], ["name", "age"]).collect() == [Row("Alice", 1)]
    assert spark.createDataFrame([1, 2, 3], T.IntegerType()).collect() == [Row(1), Row(2), Row(3)]


def test_create_pandas(spark: SparkSession) -> None:
    pd = pytest.importorskip("pandas")
    assert spark.createDataFrame(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})).collect() == [
        Row(1, "x"),
        Row(2, "y"),
    ]
    assert spark.createDataFrame(pd.DataFrame({"a": [1, 2]}), "a long").collect() == [
        Row(1),
        Row(2),
    ]
    assert spark.createDataFrame(pd.DataFrame({"a": [1, 2]}), ["x"]).collect() == [Row(1), Row(2)]


def test_create_scalars(spark: SparkSession) -> None:
    assert spark.createDataFrame([(datetime.date(2020, 1, 1),)], "d date").collect() == [Row(datetime.date(2020, 1, 1))]
    assert spark.createDataFrame([(Decimal("1.5"),)], "d decimal(10,2)").collect() == [Row(Decimal("1.50"))]
    naive = datetime.datetime(2020, 1, 1, 12, 0, 0)
    got = spark.createDataFrame([(naive,)], "t timestamp").selectExpr("CAST(t AS BIGINT) AS s").collect()
    assert got == [Row(s=int(naive.timestamp()))]
    assert spark.createDataFrame([(b"x",)], "b binary").collect() == [Row(b"x")]


def test_create_verify_schema(spark: SparkSession) -> None:
    schema = T.StructType([T.StructField("name", T.StringType(), True), T.StructField("age", T.IntegerType(), True)])
    with pytest.raises(errors.PySparkTypeError, match="can not accept"):
        spark.createDataFrame([("Alice", "x")], schema)
    strict = T.StructType([T.StructField("name", T.StringType(), False), T.StructField("age", T.IntegerType(), True)])
    with pytest.raises(errors.PySparkTypeError, match="not nullable"):
        spark.createDataFrame([(None, 1)], strict)
    # Without verification the Arrow conversion decides (a datetime truncates to a date).
    assert spark.createDataFrame(
        [(datetime.datetime(2020, 1, 2, 3, 4, 5),)], "d date", verifySchema=False
    ).collect() == [Row(datetime.date(2020, 1, 2))]
    with pytest.raises(errors.PySparkTypeError):
        spark.createDataFrame([("Alice", 1)], 1)
    assert spark.createDataFrame([("Alice", 1)]).columns == ["_1", "_2"]
    with pytest.raises(TypeError):
        spark.createDataFrame(spark.range(1))
    with pytest.raises(errors.PySparkNotImplementedError):
        spark.createDataFrame(pa.table({"a": [1]}), schema="a int")


def test_create_nested_schema_is_engine_unsupported(spark: SparkSession) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        spark.createDataFrame([([1, 2],)], T.StructType([T.StructField("a", T.ArrayType(T.IntegerType()))]))
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    with pytest.raises(errors.UnsupportedOperationException):
        spark.createDataFrame([("x",)], "a char(3)")


# session accessors ---------------------------------------------------------------


def test_empty_dataframe(spark: SparkSession, capsys: Any) -> None:
    schema = T.StructType([T.StructField("name", T.StringType(), True), T.StructField("age", T.IntegerType(), True)])
    df = spark.emptyDataFrame(schema)
    assert df.count() == 0
    df.printSchema()
    assert capsys.readouterr().out.splitlines() == [
        "root",
        " |-- name: string (nullable = true)",
        " |-- age: int (nullable = true)",
    ]
    assert spark.emptyDataFrame("name STRING, age INT").columns == ["name", "age"]
    with pytest.raises(errors.PySparkTypeError):
        spark.emptyDataFrame(1)  # ty: ignore[invalid-argument-type]


def test_catalog(spark: SparkSession) -> None:
    assert spark.catalog.currentDatabase() == "default"
    assert spark.catalog.listDatabases().collect() == [Row("default")]
    assert spark.catalog.tableExists("no_such_table") is False
    spark.range(2).createOrReplaceTempView("two")
    try:
        assert spark.catalog.tableExists("two") is True
        tables = spark.catalog.listTables().collect()
        assert [r.name for r in tables] == ["two"]
        assert tables[0].asDict() == {
            "name": "two",
            "database": "default",
            "description": None,
            "tableType": "TEMPORARY",
            "isTemporary": True,
        }
        with pytest.raises(errors.PySparkTypeError):
            spark.catalog.listTables(1)  # ty: ignore[invalid-argument-type]
    finally:
        assert spark.catalog.dropTempView("two") is True
    assert spark.catalog.tableExists("two") is False
    assert spark.catalog.dropTempView("two") is False
    with pytest.raises(errors.PySparkTypeError):
        spark.catalog.dropTempView(1)  # ty: ignore[invalid-argument-type]


def test_new_session(spark: SparkSession) -> None:
    other = spark.newSession()
    assert isinstance(other, SparkSession)
    assert other is not spark
    spark.range(1).createOrReplaceTempView("one")
    try:
        assert other.catalog.tableExists("one") is False
    finally:
        spark.catalog.dropTempView("one")


def test_create_pandas_schema_errors(spark: SparkSession) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(errors.PySparkTypeError):
        spark.createDataFrame(frame, ["a", "b"])
    with pytest.raises(errors.PySparkTypeError):
        spark.createDataFrame(frame, [1])
    with pytest.raises(errors.PySparkValueError, match="CANNOT_INFER_EMPTY_SCHEMA"):
        spark.createDataFrame([], ["a"])
    with pytest.raises(errors.PySparkTypeError):
        spark.createDataFrame([(1, 2)], [1])
    assert spark.createDataFrame([(1, 2)], ["a"]).collect() == [Row(a=1, _2=2)]


def test_create_wide_scalars(spark: SparkSession) -> None:
    assert spark.createDataFrame([(1,)], "a byte").collect() == [Row(1)]
    assert spark.createDataFrame([(1,)], "a short").collect() == [Row(1)]
    assert spark.createDataFrame([(1.5,)], "a float").collect() == [Row(1.5)]
    assert spark.createDataFrame([(1.5,)], "a double").collect() == [Row(1.5)]
    assert spark.createDataFrame([(None,)], "a void").collect() == [Row(None)]
    assert spark.createDataFrame(
        [(datetime.datetime(2020, 1, 1, 12, 0, tzinfo=datetime.UTC),)], "t timestamp_ntz"
    ).collect() == [Row(datetime.datetime(2020, 1, 1, 12, 0))]
    assert spark.createDataFrame([(None,)], "t timestamp_ntz").collect() == [Row(None)]
    with pytest.raises(errors.PySparkTypeError):
        spark.catalog.tableExists(1)  # ty: ignore[invalid-argument-type]


def test_ddl_errors() -> None:
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("`abc")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("a$")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("123")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("char x)")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("char(x)")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("`a`")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("array<int,")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("interval year to day")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("interval day to year")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("interval foo")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("a int b string")
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("a int NOT")
    assert T.DataType.fromDDL("a int NOT NULL") == T.StructType([T.StructField("a", T.IntegerType(), False)])
    assert T.DataType.fromDDL("a int NULL") == T.StructType([T.StructField("a", T.IntegerType(), True)])
    assert T.DataType.fromDDL("interval year") == T.YearMonthIntervalType(0)
    assert T.DataType.fromDDL("struct<a:int>") == T.StructType([T.StructField("a", T.IntegerType(), True)])


def test_type_reprs_and_json_values() -> None:
    assert repr(T.DayTimeIntervalType()) == "DayTimeIntervalType(0, 3)"
    assert repr(T.YearMonthIntervalType(1, 1)) == "YearMonthIntervalType(1, 1)"
    assert repr(T.ArrayType(T.StringType(), False)) == "ArrayType(StringType(), False)"
    assert repr(T.MapType(T.StringType(), T.IntegerType(), False)) == ("MapType(StringType(), IntegerType(), False)")
    assert T.VarcharType(10).jsonValue() == "varchar(10)"
    assert T._parse_datatype_json_string('"char(3)"') == T.CharType(3)


def test_create_aware_timestamp_and_no_pandas(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    # A naive datetime is read in the Python process's local time zone, as
    # PySpark reads it, never in the session time zone.
    naive = datetime.datetime(2020, 1, 1, 12, 0)
    aware = naive.astimezone(datetime.UTC)
    spark.conf.set("spark.sql.session.timeZone", "Pacific/Kiritimati")
    try:
        for schema in ("t timestamp", ["t"]):
            assert (
                spark.createDataFrame([(aware,)], schema).selectExpr("CAST(t AS BIGINT) AS s").collect()
                == spark.createDataFrame([(naive,)], schema).selectExpr("CAST(t AS BIGINT) AS s").collect()
                == [Row(s=int(aware.timestamp()))]
            )
    finally:
        spark.conf.unset("spark.sql.session.timeZone")
    # Without pandas imported, nothing is a pandas DataFrame, and the check never imports it.
    import sys

    from krishiv.sql.session import _is_pandas

    monkeypatch.delitem(sys.modules, "pandas", raising=False)
    assert _is_pandas(object()) is False
    assert "pandas" not in sys.modules


# Review follow-ups, each checked on PySpark 4.2 ------------------------------------


def test_struct_type_takes_any_iterable() -> None:
    fields = (T.StructField(n, T.IntegerType()) for n in ("a", "b"))
    assert T.StructType(fields).names == ["a", "b"]
    with pytest.raises(AssertionError, match="fields should be a list of StructField"):
        T.StructType([1])  # ty: ignore[invalid-argument-type]


def test_reversed_day_time_interval_is_rejected() -> None:
    assert T.DataType.fromDDL("interval day to hour") == T.DayTimeIntervalType(0, 1)
    with pytest.raises(errors.ParseException):
        T.DataType.fromDDL("interval hour to day")


def test_create_rows_are_positional_against_a_schema(spark: SparkSession) -> None:
    assert spark.createDataFrame([Row(b=1, a=2)], "a int, b int").collect() == [Row(a=1, b=2)]


def test_create_verify_schema_types(spark: SparkSession) -> None:
    # PySpark's accepted types: a Decimal for a decimal, a float for a double, a date or
    # datetime for a date.
    for data, schema in (([(1,)], "a decimal(10,0)"), ([(1,)], "a double")):
        with pytest.raises(errors.PySparkTypeError, match="can not accept"):
            spark.createDataFrame(data, schema)
    assert spark.createDataFrame([(datetime.datetime(2020, 1, 2, 3, 4),)], "d date").collect() == [
        Row(d=datetime.date(2020, 1, 2))
    ]


def test_create_pandas_drops_the_index(spark: SparkSession) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1, 2, 3]})
    assert spark.createDataFrame(frame[frame.a > 1]).collect() == [Row(a=2), Row(a=3)]
    assert spark.createDataFrame(frame.set_index("a", drop=False)).columns == ["a"]
    assert spark.createDataFrame(frame[frame.a > 1], ["x"]).columns == ["x"]


def test_range_spans_chunks(spark: SparkSession) -> None:
    n = (1 << 20) + 3
    assert spark.range(n).count() == n
    assert spark.range(n).filter("id = 1048578").collect() == [Row(id=n - 1)]


def test_catalog_hides_dataframe_tables(spark: SparkSession) -> None:
    import pyarrow as pa

    before = set(spark._native.temp_table_names())
    df = spark.createDataFrame(pa.table({"a": [1]}))
    # The generated name: its number comes from a counter shared by all sessions.
    [arrow_name] = set(spark._native.temp_table_names()) - before
    df.createOrReplaceTempView("visible_view")
    try:
        names = [r.name for r in spark.catalog.listTables().collect()]
        assert "visible_view" in names
        assert not any(n.lower().startswith("__krishiv_arrow_") for n in names)
        assert spark.catalog.tableExists("visible_view")
        assert not spark.catalog.tableExists(arrow_name)
        assert spark.catalog.dropTempView(arrow_name) is False
        assert spark.catalog.dropTempView(arrow_name.upper()) is False
        assert arrow_name in spark._native.temp_table_names()
        assert df.count() == 1
    finally:
        spark.catalog.dropTempView("visible_view")


def test_new_session_keeps_build_options() -> None:
    built = SparkSession.builder.config("spark.sql.session.timeZone", "Asia/Tokyo").create()
    built.conf.set("spark.sql.session.timeZone", "UTC")
    fresh = built.newSession()
    assert fresh.conf.get("spark.sql.session.timeZone") == "Asia/Tokyo"
    assert built.conf.get("spark.sql.session.timeZone") == "UTC"
