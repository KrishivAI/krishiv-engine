# SPDX-License-Identifier: Apache-2.0
"""``createDataFrame`` with array, map and struct columns: explicit schemas (the classic PySpark 4.2
verification errors), pandas and pyarrow input. Expected values are from the Spark 4.2.0 runs recorded in
``research/py-session-types-semantics.md`` (§1.2-§1.4)."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import Row, SparkSession
from krishiv.sql import types as T

pa = pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


# goldens (create_dataframe_rows.json): ddl_arr_map_struct, ddl_struct_from_row, ddl_struct_from_dict,
# ddl_dict_extra_key
def test_explicit_nested_schema(spark: SparkSession) -> None:
    ddl = "a array<int>, m map<string,double>, s struct<x:int,y:string>"
    df = spark.createDataFrame([([1, 2], {"k": 1.0}, Row(x=1, y="a")), ([3], {}, (2, "b"))], ddl)
    assert df.schema.simpleString() == "struct<a:array<int>,m:map<string,double>,s:struct<x:int,y:string>>"
    assert sorted(df.collect(), key=lambda r: r.s.x) == [
        Row(a=[1, 2], m={"k": 1.0}, s=Row(x=1, y="a")),
        Row(a=[3], m={}, s=Row(x=2, y="b")),
    ]
    # A struct column takes a dict (by field name; extra keys dropped); a dict row too.
    df = spark.createDataFrame([(1, {"x": 1, "y": "a", "z": 0})], "i int, s struct<x:int,y:string>")
    assert df.collect() == [Row(i=1, s=Row(x=1, y="a"))]
    df = spark.createDataFrame([{"a": 1, "z": 2}], "a int")
    assert df.collect() == [Row(a=1)]


# goldens: ddl_nested, ddl_null_type_arr
def test_deeply_nested_schema(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [([[1], [2, None]], {"a": [Row(p=1)]})], "a array<array<int>>, m map<string,array<struct<p:long>>>"
    )
    assert df.collect() == [Row(a=[[1], [2, None]], m={"a": [Row(p=1)]})]
    assert spark.createDataFrame([([],)], "a array<void>").collect() == [Row(a=[])]


# goldens: scalar_list_atomic_type, scalar_str_ddl
def test_atomic_schema_is_one_value_column(spark: SparkSession) -> None:
    assert spark.createDataFrame([1, 2], "int").collect() == [Row(value=1), Row(value=2)]
    assert spark.createDataFrame([[1, 2]], "array<int>").collect() == [Row(value=[1, 2])]


# goldens: ddl_int_for_str, ddl_bytes_for_str, ddl_decimal, ddl_datetime_for_date
def test_values_convert_to_the_column_type(spark: SparkSession) -> None:
    assert spark.createDataFrame([(1,)], "a string").collect() == [Row(a="1")]
    assert spark.createDataFrame([(b"ab",)], "a string").collect() == [Row(a="b'ab'")]
    assert spark.createDataFrame([(Decimal("1.235"),)], "a decimal(10,2)").collect() == [Row(a=Decimal("1.24"))]
    assert spark.createDataFrame([(datetime.datetime(2024, 1, 2, 3),)], "a date").collect() == [
        Row(a=datetime.date(2024, 1, 2))
    ]


@pytest.mark.parametrize(
    ("data", "ddl", "cls", "name"),
    [
        ([("x",)], "a int", "FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME", "field a"),
        ([(1,)], "a double", "FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME", "field a"),
        ([(1.5,)], "a int", "FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME", "field a"),
        ([(True,)], "a int", "FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME", "field a"),
        ([(1,)], "a array<int>", "FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME", "field a"),
        ([("2024-01-02",)], "a date", "FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME", "field a"),
        ([(datetime.date(2024, 1, 2),)], "a timestamp", "FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME", "field a"),
        ([(None,)], "a int not null", "FIELD_NOT_NULLABLE_WITH_NAME", "field a"),
        (
            [([None],)],
            T.StructType([T.StructField("a", T.ArrayType(T.IntegerType(), False))]),
            "FIELD_NOT_NULLABLE_WITH_NAME",
            "element in array field a",
        ),
        ([({None: 1},)], "m map<string,int>", "FIELD_NOT_NULLABLE_WITH_NAME", "key of map field m"),
        (
            [((None,),)],
            T.StructType([T.StructField("s", T.StructType([T.StructField("x", T.IntegerType(), False)]))]),
            "FIELD_NOT_NULLABLE_WITH_NAME",
            "field x in field s",
        ),
        ([(2**31,)], "a int", "VALUE_OUT_OF_BOUNDS", None),
        ([(200,)], "a tinyint", "VALUE_OUT_OF_BOUNDS", None),
        ([(1, 2)], "a int", "FIELD_STRUCT_LENGTH_MISMATCH", None),
        ([(1,)], "a int, b int", "FIELD_STRUCT_LENGTH_MISMATCH", None),
    ],
)
# goldens: ddl_str_for_int, ddl_int_for_double, ddl_float_for_int, ddl_bool_for_int, ddl_list_for_arr_wrong,
# ddl_str_for_date, ddl_date_for_ts, ddl_none_for_notnull, ddl_arr_containsNull_false, ddl_map_none_key,
# ddl_nested_notnull_struct, ddl_overflow_int, ddl_overflow_byte, ddl_too_many_fields, ddl_too_few_fields
def test_verification_errors(spark: SparkSession, data: Any, ddl: Any, cls: str, name: str | None) -> None:
    with pytest.raises((errors.PySparkTypeError, errors.PySparkValueError)) as bad:
        spark.createDataFrame(data, ddl)
    assert bad.value.getCondition() == cls
    if name is not None:
        params = bad.value.getMessageParameters()
        assert params is not None
        assert params["field_name"] == name


# golden: ddl_row_reordered_names
def test_a_row_is_read_by_position(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkTypeError) as bad:
        spark.createDataFrame([Row(b="x", a=1)], "a int, b string")
    assert str(bad.value) == (
        "[FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME] field a: IntegerType() can not accept object 'x' "
        "in type <class 'str'>."
    )


# goldens: ddl_decimal_overflow
def test_decimal_out_of_range(spark: SparkSession) -> None:
    with pytest.raises(errors.ArithmeticException) as bad:
        spark.createDataFrame([(Decimal("123456"),)], "a decimal(4,2)")
    assert bad.value.getCondition() == "NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION"
    assert bad.value.getSqlState() == "22003"


# pandas and pyarrow ---------------------------------------------------------------


# goldens (create_dataframe_pandas_arrow.json): pd_obj_list, pd_obj_dict, pd_obj_list_of_dict, pd_obj_decimal
def test_pandas_object_columns(spark: SparkSession) -> None:
    pd = pytest.importorskip("pandas")
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "l": [[1, 2], [3]],
                "d": [{"k": 1}, {"j": 2}],
                "ld": [[{"k": 1}], [{"k": 2}]],
                "n": [Decimal("1.2"), Decimal("3.456")],
            }
        )
    )
    assert df.schema.simpleString() == (
        "struct<l:array<bigint>,d:struct<k:bigint,j:bigint>,ld:array<struct<k:bigint>>,n:decimal(4,3)>"
    )
    first = df.collect()[0]
    assert first.d == Row(k=1, j=None)


# goldens: pd_with_schema, pd_with_names, pd_int_col_names
def test_pandas_with_schema_and_names(spark: SparkSession) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [[1, 2]], "m": [{"k": 1}]})
    df = spark.createDataFrame(frame, "a array<int>, m map<string,int>")
    assert df.collect() == [Row(a=[1, 2], m={"k": 1})]
    assert spark.createDataFrame(pd.DataFrame({"a": [1]}), ["z"]).columns == ["z"]
    assert spark.createDataFrame(pd.DataFrame([[1, 2]])).columns == ["0", "1"]


# goldens: pd_int32_float32, pd_nullable_int, pd_category, pd_nan_float, pd_datetime, pd_empty, pd_empty_nocols
def test_pandas_types(spark: SparkSession) -> None:
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    frame = pd.DataFrame(
        {
            "i32": np.array([1, 2], dtype="int32"),
            "f32": np.array([1.5, 2.0], dtype="float32"),
            "ni": pd.array([1, None], dtype="Int64"),
            "cat": pd.Categorical(["x", "y"]),
            "nan": [float("nan"), 1.0],
            "ts": pd.to_datetime(["2024-01-02 03:04:05", None]),
        }
    )
    assert spark.createDataFrame(frame).dtypes == [
        ("i32", "int"),
        ("f32", "float"),
        ("ni", "bigint"),
        ("cat", "string"),
        ("nan", "double"),
        ("ts", "timestamp"),
    ]
    for empty in (pd.DataFrame({"a": []}), pd.DataFrame()):
        with pytest.raises(errors.PySparkValueError) as bad:
            spark.createDataFrame(empty)
        assert bad.value.getCondition() == "CANNOT_INFER_EMPTY_SCHEMA"


# goldens: pa_large_list, pa_fixed_list, pa_map, pa_struct_nested, pa_large_string, pa_dictionary, pa_decimal256,
# pa_ts_ns
def test_pyarrow_nested_and_normalised_types(spark: SparkSession) -> None:
    table = pa.table(
        {
            "ll": pa.array([[1, 2]], pa.large_list(pa.int32())),
            "fl": pa.array([[1, 2]], pa.list_(pa.int32(), 2)),
            "m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int64())),
            "s": [{"x": [1, 2], "y": {"z": 1.5}}],
            "ls": pa.array(["x"], pa.large_string()),
            "dic": pa.array(["x"]).dictionary_encode(),
            "d256": pa.array([Decimal("1.20")], pa.decimal256(10, 2)),
            "ns": pa.array([datetime.datetime(2024, 1, 2, 3)], pa.timestamp("ns")),
        }
    )
    df = spark.createDataFrame(table)
    assert df.schema.simpleString() == (
        "struct<ll:array<int>,fl:array<int>,m:map<string,bigint>,s:struct<x:array<bigint>,y:struct<z:double>>,"
        "ls:string,dic:string,d256:decimal(10,2),ns:timestamp>"
    )
    row = df.first()
    assert row is not None
    assert (row.ll, row.m, row.s, row.ns) == (
        [1, 2],
        {"k": 1},
        Row(x=[1, 2], y=Row(z=1.5)),
        datetime.datetime(2024, 1, 2, 3),
    )


@pytest.mark.parametrize(
    "column",
    [
        lambda: pa.array([1], pa.uint8()),
        lambda: pa.array([1], pa.uint64()),
        lambda: pa.array([1.0], pa.float32()).cast(pa.float16()),
        lambda: pa.array([datetime.date(2024, 1, 2)], pa.date64()),
    ],
)
def test_pyarrow_types_spark_lacks(spark: SparkSession, column: Any) -> None:
    with pytest.raises(errors.PySparkTypeError) as bad:
        spark.createDataFrame(pa.table({"a": column()}))
    assert bad.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"


# goldens: pa_with_schema_cast, pa_with_names, pa_empty (classic keeps the schema)
def test_pyarrow_with_schema_names_and_empty(spark: SparkSession) -> None:
    assert spark.createDataFrame(pa.table({"a": [1, 2]}), "a string").collect() == [Row(a="1"), Row(a="2")]
    assert spark.createDataFrame(pa.table({"a": [1]}), ["z"]).columns == ["z"]
    assert spark.createDataFrame(pa.table({"a": pa.array([], pa.int64())})).dtypes == [("a", "bigint")]


# ddl_nested_notnull_struct-style nested path; the message is PySpark 4.2.0 classic's (review run)
def test_nested_struct_length_mismatch_names_the_field(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkValueError) as bad:
        spark.createDataFrame([((1,),)], "s struct<x:int,y:int>")
    assert bad.value.getCondition() == "FIELD_STRUCT_LENGTH_MISMATCH_WITH_NAME"
    assert str(bad.value) == (
        "[FIELD_STRUCT_LENGTH_MISMATCH_WITH_NAME] field s: Length of object (1) "
        "does not match with length of fields (2)."
    )


# goldens: ddl_verifySchema_false
def test_verify_schema_false_gives_null(spark: SparkSession) -> None:
    assert spark.createDataFrame([("x",), (1,)], "a int", verifySchema=False).collect() == [
        Row(a=None),
        Row(a=1),
    ]


# goldens: rows_mismatched_fields, infer_struct_diff_fields, dict_keys_sorted
def test_rows_with_different_fields_do_not_shift_values(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkValueError) as bad:
        spark.createDataFrame([Row(a=1), Row(b=2)])
    assert bad.value.getCondition() == "STRUCT_ARRAY_LENGTH_MISMATCH"
    assert bad.value.getSqlState() == "2201E"
    # A dict row is read by its sorted keys, then the names rename by position.
    assert spark.createDataFrame([{"z": 1, "a": 2}], ["z", "a"]).collect() == [Row(z=2, a=1)]
    assert spark.createDataFrame([{"a": 1}, {"b": 2}]).collect() == [Row(a=1, b=None), Row(a=None, b=2)]


# golden: names_dup (Spark keeps both columns; the engine reads registered tables by name, so it is refused)
def test_duplicate_column_names_are_refused(spark: SparkSession) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        spark.createDataFrame([(1, 2)], ["a", "a"])
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert spark.createDataFrame([(1, 2)], ["a", "b"]).columns == ["a", "b"]


# goldens: array_array (typecode u is a string)
def test_array_array_of_text(spark: SparkSession) -> None:
    import array

    frame = spark.createDataFrame([(array.array("u", "ab"),)])  # ty: ignore[deprecated]
    assert frame.dtypes == [("_1", "array<string>")]


def test_struct_schema_rejects_a_scalar_row(spark: SparkSession) -> None:
    with pytest.raises(errors.PySparkTypeError) as bad:
        spark.createDataFrame([1], "struct<a:int>")
    assert bad.value.getCondition() == "FIELD_DATA_TYPE_UNACCEPTABLE"


def test_naive_timestamps_in_a_dst_gap(spark: SparkSession) -> None:
    pd = pytest.importorskip("pandas")
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    try:
        frame = pd.DataFrame({"t": pd.to_datetime(["2024-03-10 02:30:00", "2024-11-03 01:30:00"])})
        assert spark.createDataFrame(frame).count() == 2
    finally:
        spark.conf.unset("spark.sql.session.timeZone")
