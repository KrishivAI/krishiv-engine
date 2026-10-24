# SPDX-License-Identifier: Apache-2.0
"""``krishiv.testing``, ported from PySpark 4.2's ``test_utils.py``, plus error-accessor tests.

Only imports and session setup are adapted, except where the engine differs (noted):
no ``range``/temp views/``selectExpr``/``regexp_replace``/``to_date`` (those cases use
``createDataFrame`` rows or plain SQL instead); nested frames use lists of Rows (the
engine has no nested types yet); duplicate-column frames use lists of Rows (the engine
rejects duplicate names); pandas cases skip when pandas is missing; pandas-on-Spark,
streaming and ``TimestampNTZNanosType`` cases have no equivalent here.
"""

from __future__ import annotations

import difflib
from decimal import Decimal

import pytest

from krishiv import errors
from krishiv.errors import QueryContextType
from krishiv.sql import Row, SparkSession
from krishiv.sql import types as T
from krishiv.testing.utils import _context_diff, assertDataFrameEqual, assertSchemaEqual

pa = pytest.importorskip("pyarrow")

orders = pytest.mark.parametrize("check_order", [False, True])


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def rows_error(actual: list[Row], expected: list[Row], count: int, total: int) -> dict[str, str]:
    first = "".join(str(r) + "\n" for r in actual)
    second = "".join(str(r) + "\n" for r in expected)
    diff = _context_diff(actual=first.splitlines(), expected=second.splitlines(), n=total)
    return {"error_msg": f"Results do not match: ( {(count / total) * 100:.5f} % )\n" + "\n".join(diff)}


@orders
def test_assert_equal_inttype(spark: SparkSession, check_order: bool) -> None:
    df1 = spark.createDataFrame(data=[("1", 1000), ("2", 3000)], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[("1", 1000), ("2", 3000)], schema=["id", "amount"])
    assertDataFrameEqual(df1, df2, checkRowOrder=check_order)


@orders
def test_assert_equal_arraytype(check_order: bool) -> None:
    rows = [Row(name="john", languages=["Python", "Java"]), Row(name="jane", languages=["S", "Q", "J"])]
    assertDataFrameEqual(list(rows), list(rows), checkRowOrder=check_order)


@orders
def test_assert_approx_equal_arraytype_float(check_order: bool) -> None:
    rows1 = [Row(student="s1", grades=[97.01, 89.23]), Row(student="s2", grades=[91.86, 84.34])]
    rows2 = [Row(student="s1", grades=[97.01, 89.23]), Row(student="s2", grades=[91.86, 84.339999])]
    assertDataFrameEqual(rows1, rows2, checkRowOrder=check_order)


@orders
def test_assert_approx_equal_arraytype_float_default_rtol_fail(check_order: bool) -> None:
    rows1 = [Row(student="s1", grades=[97.01, 89.23]), Row(student="s2", grades=[91.86, 84.34])]
    rows2 = [Row(student="s1", grades=[97.01, 89.23]), Row(student="s2", grades=[91.86, 84.341])]
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(rows1, rows2, checkRowOrder=check_order)
    assert pe.value.getCondition() == "DIFFERENT_ROWS"
    assert pe.value.getMessageParameters() == rows_error(rows1, rows2, 1, 2)


def test_assert_approx_equal_arraytype_float_custom_rtol_pass() -> None:
    rows1 = [Row(student="s1", grades=[97.01, 89.23]), Row(student="s2", grades=[91.86, 84.34])]
    rows2 = [Row(student="s1", grades=[97.01, 89.23]), Row(student="s2", grades=[91.86, 84.341])]
    assertDataFrameEqual(rows1, rows2, rtol=1e-2)


def test_assert_approx_equal_decimaltype_custom_rtol_pass(spark: SparkSession) -> None:
    schema = T.StructType([T.StructField("grade", T.DecimalType(5, 3), True)])
    df1 = spark.createDataFrame(data=[(Decimal("83.140"),), (Decimal("97.120"),)], schema=schema)
    df2 = spark.createDataFrame(data=[(Decimal("83.140"),), (Decimal("97.111"),)], schema=schema)
    assertDataFrameEqual(df1, df2, rtol=1e-1)


def test_assert_notequal_arraytype() -> None:
    rows1 = [Row(name="Amy", languages=["C++", "Rust"]), Row(name="John", languages=["Py", "J"])]
    rows2 = [Row(name="Amy", languages=["C++", "Rust"]), Row(name="John", languages=["Py", "X"])]
    ordered = (sorted(rows1, key=str), sorted(rows2, key=str))
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(rows1, rows2)
    assert pe.value.getCondition() == "DIFFERENT_ROWS"
    assert pe.value.getMessageParameters() == rows_error(*ordered, 1, 2)


def test_assert_equal_maptype() -> None:
    rows = [Row(student="s1", properties={"id": 1}), Row(student="s2", properties={"id": 2})]
    assertDataFrameEqual(list(rows), list(rows), checkRowOrder=False)
    assertDataFrameEqual(list(rows), list(rows), checkRowOrder=True)


def test_assert_approx_equal_nested_struct_double() -> None:
    rows1 = [Row(name="jane", grades=Row(math=64.57, english=76.63))]
    rows2 = [Row(name="jane", grades=Row(math=64.57, english=76.63000001))]
    assertDataFrameEqual(rows1, rows2, checkRowOrder=False)
    assertDataFrameEqual(rows1, rows2, checkRowOrder=True)


def test_assert_equal_duplicate_col() -> None:
    # The engine rejects duplicate column names, so this runs on lists of Rows.
    rows = [Row(1, "Python", 1, 1), Row(2, "Scala", 2, 2)]
    assertDataFrameEqual(list(rows), list(rows), checkRowOrder=False)
    assertDataFrameEqual(list(rows), list(rows), checkRowOrder=True)


@orders
def test_assert_equal_nullrow(spark: SparkSession, check_order: bool) -> None:
    df1 = spark.createDataFrame(data=[("1", 1000), (None, None)], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[("1", 1000), (None, None)], schema=["id", "amount"])
    assertDataFrameEqual(df1, df2, checkRowOrder=check_order)


@orders
def test_assert_notequal_nullval(spark: SparkSession, check_order: bool) -> None:
    df1 = spark.createDataFrame(data=[("1", 1000), ("2", 2000)], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[("1", 1000), ("2", None)], schema=["id", "amount"])
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(df1, df2, checkRowOrder=check_order)
    assert pe.value.getCondition() == "DIFFERENT_ROWS"
    assert pe.value.getMessageParameters() == rows_error(df1.collect(), df2.collect(), 1, 2)


@orders
def test_assert_equal_nulldf(check_order: bool) -> None:
    assertDataFrameEqual(None, None, checkRowOrder=check_order)


def test_assert_unequal_null_frames(spark: SparkSession) -> None:
    df = spark.createDataFrame(data=[("1", 1000), ("2", 3000)], schema=["id", "amount"])
    with pytest.raises(errors.PySparkAssertionError) as actual:
        assertDataFrameEqual(None, df)
    assert actual.value.getCondition() == "INVALID_TYPE_DF_EQUALITY_ARG"
    assert actual.value.getMessageParameters() == {
        "expected_type": "Union[DataFrame, ps.DataFrame, List[Row]]",
        "arg_name": "actual",
        "actual_type": None,
    }
    with pytest.raises(errors.PySparkAssertionError) as expected:
        assertDataFrameEqual(df, None)
    expected_params = expected.value.getMessageParameters()
    assert expected_params is not None
    assert expected_params["arg_name"] == "expected"


def test_assert_error_non_pyspark_df() -> None:
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual({"a": 1}, {"a": 1})
    assert pe.value.getCondition() == "INVALID_TYPE_DF_EQUALITY_ARG"
    error_params = pe.value.getMessageParameters()
    assert error_params is not None
    assert error_params["arg_name"] == "actual"
    assert error_params["actual_type"] is dict


def test_row_order_ignored(spark: SparkSession) -> None:
    df1 = spark.createDataFrame(data=[("2", 3000.00), ("1", 1000.00)], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[("1", 1000.00), ("2", 3000.00)], schema=["id", "amount"])
    assertDataFrameEqual(df1, df2)


def test_check_row_order_error(spark: SparkSession) -> None:
    df1 = spark.createDataFrame(data=[("2", 3000.00), ("1", 1000.00)], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[("1", 1000.00), ("2", 3000.00)], schema=["id", "amount"])
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(df1, df2, checkRowOrder=True)
    assert pe.value.getMessageParameters() == rows_error(df1.collect(), df2.collect(), 2, 2)


def test_different_row_count_no_cascading_diff() -> None:
    # A row missing from the middle counts once; later rows still pair up.
    actual = [Row(id="1"), Row(id="2"), Row(id="3"), Row(id="4"), Row(id="5")]
    expected = [Row(id="1"), Row(id="2"), Row(id="4"), Row(id="5")]
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(actual, expected)
    no_cascade_params = pe.value.getMessageParameters()
    assert no_cascade_params is not None
    assert "20.00000 %" in no_cascade_params["error_msg"]
    with pytest.raises(errors.PySparkAssertionError) as include:
        assertDataFrameEqual(actual, expected, includeDiffRows=True)
    assert include.value.data == [(Row(id="3"), None)]


def test_different_row_count_mixed_extra_and_missing() -> None:
    actual = [Row(v="A"), Row(v="B"), Row(v="D"), Row(v="X")]
    expected = [Row(v="A"), Row(v="B"), Row(v="C"), Row(v="D"), Row(v="E")]
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(actual, expected, includeDiffRows=True)
    assert pe.value.data == [(None, Row(v="C")), (None, Row(v="E")), (Row(v="X"), None)]
    mixed_params = pe.value.getMessageParameters()
    assert mixed_params is not None
    assert "60.00000 %" in mixed_params["error_msg"]


@orders
def test_assert_pyspark_approx_equal(spark: SparkSession, check_order: bool) -> None:
    df1 = spark.createDataFrame(data=[("1", 1000.00), ("2", 3000.00)], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[("1", 1000.0000001), ("2", 3000.00)], schema=["id", "amount"])
    assertDataFrameEqual(df1, df2, checkRowOrder=check_order)


@orders
def test_assert_pyspark_df_not_equal(spark: SparkSession, check_order: bool) -> None:
    df1 = spark.createDataFrame(data=[("1", 1000.00), ("2", 3000.00), ("3", 2000.00)], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[("1", 1001.00), ("2", 3000.00), ("3", 2003.00)], schema=["id", "amount"])
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(df1, df2, checkRowOrder=check_order)
    assert pe.value.getMessageParameters() == rows_error(df1.collect(), df2.collect(), 2, 3)


def test_assert_notequal_schema(spark: SparkSession) -> None:
    df1 = spark.createDataFrame(data=[(1, 1000), (2, 3000)], schema=["id", "number"])
    df2 = spark.createDataFrame(data=[("1", 1000), ("2", 5000)], schema=["id", "amount"])
    diff = difflib.ndiff(str(df1.schema).splitlines(), str(df2.schema).splitlines())
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(df1, df2)
    assert pe.value.getCondition() == "DIFFERENT_SCHEMA"
    assert pe.value.getMessageParameters() == {"error_msg": "\n".join(diff)}


def test_schema_ignore_nullable(spark: SparkSession) -> None:
    s1 = T.StructType([T.StructField("id", T.IntegerType(), True), T.StructField("name", T.StringType(), True)])
    s2 = T.StructType([T.StructField("id", T.IntegerType(), True), T.StructField("name", T.StringType(), False)])
    df1 = spark.createDataFrame([(1, "jane"), (2, "john")], s1)
    df2 = spark.createDataFrame([(1, "jane"), (2, "john")], s2)
    assertDataFrameEqual(df1, df2)
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual(df1, df2, ignoreNullable=False)


def test_schema_ignore_nullable_collections() -> None:
    assertSchemaEqual(
        T.StructType([T.StructField("names", T.ArrayType(T.DoubleType(), True), True)]),
        T.StructType([T.StructField("names", T.ArrayType(T.DoubleType(), False), False)]),
    )
    with pytest.raises(errors.PySparkAssertionError):
        assertSchemaEqual(
            T.StructType([T.StructField("m", T.MapType(T.StringType(), T.IntegerType()), True)]),
            T.StructType([T.StructField("m", T.MapType(T.IntegerType(), T.StringType()), True)]),
            ignoreNullable=True,
        )
    assertSchemaEqual(
        T.StructType([T.StructField("n", T.StructType([T.StructField("age", T.IntegerType(), True)]), True)]),
        T.StructType([T.StructField("n", T.StructType([T.StructField("age", T.IntegerType(), False)]), False)]),
    )


def test_schema_collection_unequal() -> None:
    s1 = T.StructType([T.StructField("names", T.ArrayType(T.IntegerType(), True), True)])
    s2 = T.StructType([T.StructField("names", T.ArrayType(T.DoubleType(), False), False)])
    diff = difflib.ndiff(str(s1).splitlines(), str(s2).splitlines())
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertSchemaEqual(s1, s2)
    assert pe.value.getCondition() == "DIFFERENT_SCHEMA"
    assert pe.value.getMessageParameters() == {"error_msg": "\n".join(diff)}
    with pytest.raises(errors.PySparkAssertionError):
        assertSchemaEqual(
            T.StructType([T.StructField("n", T.StructType([T.StructField("age", T.DoubleType(), True)]), True)]),
            T.StructType([T.StructField("n", T.StructType([T.StructField("age", T.IntegerType(), True)]), True)]),
        )


def test_schema_unsupported_type() -> None:
    with pytest.raises(errors.PySparkTypeError) as pe:
        assertSchemaEqual("names: int", "names: int")
    assert pe.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert pe.value.getMessageParameters() == {
        "arg_name": "actual",
        "expected_type": "struct type",
        "arg_type": "str",
    }


def test_schema_with_decimal_types() -> None:
    s1 = T.StructType([T.StructField("price", T.DecimalType(10, 2), True)])
    assertSchemaEqual(s1, T.StructType([T.StructField("price", T.DecimalType(10, 2), True)]))
    with pytest.raises(errors.PySparkAssertionError):
        assertSchemaEqual(s1, T.StructType([T.StructField("price", T.DecimalType(12, 4), True)]))


def test_spark_sql(spark: SparkSession) -> None:
    assertDataFrameEqual(spark.sql("select 1 + 2 AS x"), spark.sql("select 3 AS x"))
    assertDataFrameEqual(spark.sql("select 1 + 2 AS x"), spark.sql("select 3 AS x"), checkRowOrder=True)


def test_spark_sql_sort_rows(spark: SparkSession) -> None:
    df1 = spark.createDataFrame(data=[(1, 3000), (2, 1000)], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[(2, 1000), (1, 3000)], schema=["id", "amount"])
    assertDataFrameEqual(df1.orderBy("amount"), df2)
    assertDataFrameEqual(df1.orderBy("amount"), df2, checkRowOrder=True)


def test_empty_dataset(spark: SparkSession) -> None:
    df1 = spark.createDataFrame(data=[], schema="id INT, amount DOUBLE")
    df2 = spark.createDataFrame(data=[], schema="id INT, amount DOUBLE")
    assertDataFrameEqual(df1, df2, checkRowOrder=False)
    assertDataFrameEqual(df1, df2, checkRowOrder=True)


def test_no_column(spark: SparkSession) -> None:
    df1 = spark.createDataFrame(pa.table({"id": [1, 2]})).select()
    df2 = spark.createDataFrame(pa.table({"id": [1, 2]})).select()
    assertDataFrameEqual(df1, df2, checkRowOrder=False)
    assertDataFrameEqual(df1, df2, checkRowOrder=True)


def test_empty_expected_list(spark: SparkSession) -> None:
    df = spark.createDataFrame(pa.table({"id": [1, 2]})).select()
    assertDataFrameEqual(df, [Row(), Row()], checkRowOrder=False)
    assertDataFrameEqual(df, [Row(), Row()], checkRowOrder=True)


@orders
def test_special_vals(spark: SparkSession, check_order: bool) -> None:
    df1 = spark.createDataFrame(data=[(1, float("nan")), (2, float("inf"))], schema=["id", "amount"])
    df2 = spark.createDataFrame(data=[(1, float("nan")), (2, float("inf"))], schema=["id", "amount"])
    assertDataFrameEqual(df1, df2, checkRowOrder=check_order)


@orders
def test_special_vals_not_equal(check_order: bool) -> None:
    nan_rows = [Row(id=1, amount=float("nan"))]
    finite_rows = [Row(id=1, amount=5.0)]
    with pytest.raises(errors.PySparkAssertionError) as nan_vs_finite:
        assertDataFrameEqual(nan_rows, finite_rows, checkRowOrder=check_order)
    assert nan_vs_finite.value.getCondition() == "DIFFERENT_ROWS"
    assert nan_vs_finite.value.getMessageParameters() == rows_error(nan_rows, finite_rows, 1, 1)

    finite = [Row(id=1, amount=5.0)]
    infinite = [Row(id=1, amount=float("inf"))]
    with pytest.raises(errors.PySparkAssertionError) as finite_vs_inf:
        assertDataFrameEqual(finite, infinite, checkRowOrder=check_order)
    assert finite_vs_inf.value.getMessageParameters() == rows_error(finite, infinite, 1, 1)

    pos_inf = [Row(id=1, amount=float("inf"))]
    neg_inf = [Row(id=1, amount=float("-inf"))]
    with pytest.raises(errors.PySparkAssertionError) as inf_vs_neg_inf:
        assertDataFrameEqual(pos_inf, neg_inf, checkRowOrder=check_order)
    assert inf_vs_neg_inf.value.getMessageParameters() == rows_error(pos_inf, neg_inf, 1, 1)


def test_decimal_special_vals() -> None:
    nan = [Row(v=Decimal("NaN"))]
    assertDataFrameEqual(nan, [Row(v=Decimal("NaN"))])
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual(nan, [Row(v=Decimal("1.0"))])
    assertDataFrameEqual([Row(v=Decimal("Infinity"))], [Row(v=Decimal("Infinity"))])
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual([Row(v=Decimal("Infinity"))], [Row(v=Decimal("1.0"))])
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual([Row(v=Decimal("Infinity"))], [Row(v=Decimal("-Infinity"))])


@orders
def test_df_list_row_equal(spark: SparkSession, check_order: bool) -> None:
    df1 = spark.createDataFrame(data=[(1, 3000), (2, 1000)], schema=["id", "amount"])
    assertDataFrameEqual(df1, [Row(1, 3000), Row(2, 1000)], checkRowOrder=check_order)


@orders
def test_list_rows_equal(check_order: bool) -> None:
    assertDataFrameEqual(
        [Row(1, "abc", 5000), Row(2, "def", 1000)],
        [Row(1, "abc", 5000), Row(2, "def", 1000)],
        checkRowOrder=check_order,
    )


@orders
def test_list_rows_unequal(check_order: bool) -> None:
    rows1 = [Row(1, "abc", 5000), Row(2, "def", 1000)]
    rows2 = [Row(1, "abc", 5000), Row(2, "defg", 1000)]
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(rows1, rows2, checkRowOrder=check_order)
    assert pe.value.getCondition() == "DIFFERENT_ROWS"
    assert pe.value.getMessageParameters() == rows_error(rows1, rows2, 1, 2)


def test_list_row_unequal_schema(spark: SparkSession) -> None:
    df1 = spark.createDataFrame(data=[(1, 3000), (2, 1000)], schema=["id", "amount"])
    rows2 = [Row(1, "3000"), Row(2, "1000")]
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(df1, rows2)
    assert pe.value.getMessageParameters() == rows_error(df1.collect(), rows2, 2, 2)


def test_dataframe_include_diff_rows(spark: SparkSession) -> None:
    df1 = spark.createDataFrame([("1", 1000.00), ("2", 3000.00), ("3", 2000.00)], ["id", "amount"])
    df2 = spark.createDataFrame([("1", 1001.00), ("2", 3000.00), ("3", 2003.00)], ["id", "amount"])
    with pytest.raises(errors.PySparkAssertionError) as ctx:
        assertDataFrameEqual(df1, df2, includeDiffRows=True)
    assert ctx.value.data == [
        (Row(id="1", amount=1000.0), Row(id="1", amount=1001.0)),
        (Row(id="3", amount=2000.0), Row(id="3", amount=2003.0)),
    ]


def test_dataframe_ignore_column_order(spark: SparkSession) -> None:
    df1 = spark.createDataFrame([Row(A=1, B=2), Row(A=3, B=4)])
    df2 = spark.createDataFrame([Row(B=2, A=1), Row(B=4, A=3)])
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual(df1, df2, ignoreColumnOrder=False)
    assertDataFrameEqual(df1, df2, ignoreColumnOrder=True)


def test_dataframe_ignore_column_name(spark: SparkSession) -> None:
    df1 = spark.createDataFrame([(1, 2), (3, 4)], ["A", "B"])
    df2 = spark.createDataFrame([(1, 2), (3, 4)], ["X", "Y"])
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual(df1, df2, ignoreColumnName=False)
    assertDataFrameEqual(df1, df2, ignoreColumnName=True)


def test_dataframe_ignore_column_type(spark: SparkSession) -> None:
    df1 = spark.createDataFrame([(1, "2"), (3, "4")], ["A", "B"])
    df2 = spark.createDataFrame([(1, 2), (3, 4)], ["A", "B"])
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual(df1, df2, ignoreColumnType=False)
    assertDataFrameEqual(df1, df2, ignoreColumnType=True)


def test_ignore_column_type_bool_renders_lowercase(spark: SparkSession) -> None:
    df_true = spark.createDataFrame([(True,)], ["flag"])
    df_lower = spark.createDataFrame([("true",)], ["flag"])
    df_upper = spark.createDataFrame([("True",)], ["flag"])
    assertDataFrameEqual(df_true, df_lower, ignoreColumnType=True)
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual(df_true, df_upper, ignoreColumnType=True)


def test_ignore_column_type_float_renders_like_cast(spark: SparkSession) -> None:
    df_float = spark.createDataFrame([(1.0,)], ["v"])
    df_str = spark.createDataFrame([("1.0",)], ["v"])
    df_one = spark.createDataFrame([("1",)], ["v"])
    assertDataFrameEqual(df_float, df_str, ignoreColumnType=True)
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual(df_float, df_one, ignoreColumnType=True)
    assertDataFrameEqual(
        spark.createDataFrame([(1e16,)], ["v"]), spark.createDataFrame([("1.0E16",)], ["v"]), ignoreColumnType=True
    )
    assertDataFrameEqual(
        spark.createDataFrame([(1.5e-07,)], ["v"]), spark.createDataFrame([("1.5E-7",)], ["v"]), ignoreColumnType=True
    )


def test_dataframe_max_errors(spark: SparkSession) -> None:
    df1 = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c"), (4, "d")], ["id", "value"])
    df2 = spark.createDataFrame([(1, "a"), (2, "z"), (3, "x"), (4, "y")], ["id", "value"])
    with pytest.raises(errors.PySparkAssertionError) as ctx:
        assertDataFrameEqual(df1, df2, maxErrors=2)
    assert str(ctx.value).count("! Row") == 4


def test_dataframe_show_only_diff(spark: SparkSession) -> None:
    df1 = spark.createDataFrame([(1, "apple", "red"), (2, "banana", "yellow"), (3, "cherry", "red")], ["i", "f", "c"])
    df2 = spark.createDataFrame(
        [(1, "apple", "green"), (2, "banana", "yellow"), (3, "cherry", "blue")], ["i", "f", "c"]
    )
    with pytest.raises(errors.PySparkAssertionError) as ctx:
        assertDataFrameEqual(df1, df2, showOnlyDiff=False)
    assert "apple" in str(ctx.value)
    assert "banana" in str(ctx.value)
    with pytest.raises(errors.PySparkAssertionError) as diff:
        assertDataFrameEqual(df1, df2, showOnlyDiff=True)
    assert "apple" in str(diff.value)
    assert "banana" not in str(diff.value)


def test_assert_equal_exact_pandas_df() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    df1 = pd.DataFrame(data=np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)]), columns=["a", "b", "c"])
    df2 = pd.DataFrame(data=np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)]), columns=["a", "b", "c"])
    assertDataFrameEqual(df1, df2, checkRowOrder=False)
    assertDataFrameEqual(df1, df2, checkRowOrder=True)


def test_assert_approx_equal_pandas_df() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    df1 = pd.DataFrame(data=np.array([(1, 2, 3), (4, 5, 6), (7, 8, 59)]), columns=["a", "b", "c"])
    df2 = pd.DataFrame(data=np.array([(1, 2, 3), (4, 5, 6), (7, 8, 59.0001)]), columns=["a", "b", "c"])
    assertDataFrameEqual(df1, df2, checkRowOrder=False)
    assertDataFrameEqual(df1, df2, checkRowOrder=True)


def test_assert_unequal_pandas_df() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    df1 = pd.DataFrame(data=np.array([(1, 2, 3), (4, 5, 6), (6, 5, 4)]), columns=["a", "b", "c"])
    df2 = pd.DataFrame(data=np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)]), columns=["a", "b", "c"])
    with pytest.raises(errors.PySparkAssertionError) as pe:
        assertDataFrameEqual(df1, df2, checkRowOrder=False)
    assert pe.value.getCondition() == "DIFFERENT_PANDAS_DATAFRAME"
    pandas_params = pe.value.getMessageParameters()
    assert pandas_params is not None
    assert set(pandas_params) == {"left", "left_dtype", "right", "right_dtype"}


def test_capture_parse_exception(spark: SparkSession) -> None:
    with pytest.raises(errors.ParseException):
        spark.sql("abc")


def test_capture_analysis_exception(spark: SparkSession) -> None:
    with pytest.raises(errors.AnalysisException):
        spark.sql("select abc")


def test_divide_by_zero_accessors(spark: SparkSession) -> None:
    df = spark.sql("select 1 / 0 as x")
    with pytest.raises(errors.ArithmeticException) as raised:
        df.collect()
    e = raised.value
    assert e.getCondition() == "DIVIDE_BY_ZERO"
    assert e.getSqlState() == "22012"
    assert e.getMessageParameters() == {"config": '"spark.sql.ansi.enabled"'}
    assert e.getMessage() == str(e)
    assert "[DIVIDE_BY_ZERO]" in e.getMessage()
    # Divergence from Spark 4.2: the engine attaches no query context to
    # execution errors, so this is [] here (Spark would carry the failed node).
    assert e.getQueryContext() == []


def test_unresolved_column_sql_context(spark: SparkSession) -> None:
    with pytest.raises(errors.AnalysisException) as raised:
        spark.sql("SELECT a")
    e = raised.value
    assert e.getCondition() == "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    assert e.getSqlState() == "42703"
    assert e.getMessageParameters() == {"objectName": "`a`"}
    assert len(e.getQueryContext()) == 1
    qc = e.getQueryContext()[0]
    assert qc.contextType() == QueryContextType.SQL
    assert qc.fragment() == "a"
    assert qc.startIndex() == 7
    assert qc.stopIndex() == 7
    assert qc.objectName() == ""
    assert qc.objectType() == ""


def test_unresolved_column_dataframe_context(spark: SparkSession) -> None:
    df = spark.createDataFrame(pa.table({"a": [1, 2]}))
    with pytest.raises(errors.AnalysisException) as raised:
        df.select("missing").collect()
    (qc,) = raised.value.getQueryContext()
    assert qc.contextType() == QueryContextType.DataFrame


def test_get_error_class_is_deprecated(spark: SparkSession) -> None:
    with pytest.raises(errors.AnalysisException) as raised:
        spark.sql("SELECT a")
    e = raised.value
    with pytest.warns(FutureWarning, match="getCondition"):
        assert e.getErrorClass() == e.getCondition()


def test_query_context_accessors() -> None:
    sql_ctx = errors.SqlQueryContext(
        fragment="a", start=7, stop=7, objectType="TABLE", objectName="t", callSite="site", summary="sum"
    )
    assert sql_ctx.contextType() == QueryContextType.SQL
    assert sql_ctx.fragment() == "a"
    assert sql_ctx.startIndex() == 7
    assert sql_ctx.stopIndex() == 7
    assert sql_ctx.objectType() == "TABLE"
    assert sql_ctx.objectName() == "t"
    assert sql_ctx.callSite() == "site"
    assert sql_ctx.summary() == "sum"

    df_ctx = errors.DataFrameQueryContext(callSite="site", summary="sum")
    assert df_ctx.contextType() == QueryContextType.DataFrame
    assert df_ctx.objectType() == ""
    assert df_ctx.objectName() == ""
    assert df_ctx.startIndex() == 0
    assert df_ctx.stopIndex() == 0
    assert df_ctx.fragment() == ""
    assert df_ctx.callSite() == "site"
    assert df_ctx.summary() == "sum"


def test_collection_type_simple_strings() -> None:
    assert T.ArrayType(T.DoubleType(), True).simpleString() == "array<double>"
    assert T.MapType(T.StringType(), T.IntegerType()).simpleString() == "map<string,int>"


def test_create_dataframe_rejects_bad_shapes(spark: SparkSession) -> None:
    # More names than row values fail; fewer keep the inferred `_n` names (Spark 4.2).
    assert spark.createDataFrame([(1, 2, 3)], ["a", "b"]).columns == ["a", "b", "_3"]
    with pytest.raises(
        errors.PySparkValueError, match=r"\[AXIS_LENGTH_MISMATCH\] .* axis has 2 elements, new values have 1"
    ):
        spark.createDataFrame([(1,)], ["a", "b"])
    with pytest.raises(
        errors.PySparkValueError, match=r"\[AXIS_LENGTH_MISMATCH\] .* axis has 2 elements, new values have 1"
    ):
        spark.createDataFrame([(1, 2), (3,)], ["a", "b"])
    with pytest.raises(errors.PySparkValueError, match="CANNOT_INFER_EMPTY_SCHEMA"):
        spark.createDataFrame([], None)
    with pytest.raises(errors.PySparkTypeError):
        spark.createDataFrame([(1,)], schema=123)
    with pytest.raises(errors.PySparkTypeError):
        spark.createDataFrame([(1,)], schema=[1])
    with pytest.raises(errors.PySparkTypeError):
        spark.createDataFrame([1], schema=["a"])
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        spark.createDataFrame({"a": 1})
    with pytest.raises(errors.UnsupportedOperationException, match="KRISHIV_UNSUPPORTED_FEATURE"):
        spark.createDataFrame(
            [(1,)],
            T.StructType([T.StructField("a", T.ArrayType(T.IntegerType()), True)]),
        )


def test_create_dataframe_dict_missing_is_null(spark: SparkSession) -> None:
    df = spark.createDataFrame([{"a": 1}, {"b": 2}], ["a", "b"])
    assert df.columns == ["a", "b"]
    assert df.collect() == [Row(a=1, b=None), Row(a=None, b=2)]


def test_create_dataframe_infers_named_and_positional(spark: SparkSession) -> None:
    named = spark.createDataFrame([Row(a=1, b="x"), Row(a=2, b="y")])
    assert named.columns == ["a", "b"]
    by_dict = spark.createDataFrame([{"a": 1}, {"a": 2}])
    assert by_dict.columns == ["a"]
    positional = spark.createDataFrame([(1, 2)])
    assert positional.columns == ["_1", "_2"]


def test_create_dataframe_infers_timestamp_type(spark: SparkSession) -> None:
    from datetime import datetime

    df = spark.createDataFrame([(datetime(2024, 1, 2, 3, 4, 5),)], ["ts"])
    assert df.dtypes == [("ts", "timestamp")]
    assert df.collect() == [Row(ts=datetime(2024, 1, 2, 3, 4, 5))]
    ntz = spark.createDataFrame(
        [(datetime(2024, 1, 2, 3, 4, 5),)],
        T.StructType([T.StructField("ts", T.TimestampNTZType(), True)]),
    )
    assert ntz.dtypes == [("ts", "timestamp_ntz")]
    explicit = spark.createDataFrame(
        [(datetime(2024, 1, 2, 3, 4, 5),)],
        T.StructType([T.StructField("ts", T.TimestampType(), True)]),
    )
    assert explicit.dtypes == [("ts", "timestamp")]


def test_schema_ignores_order_and_names(spark: SparkSession) -> None:
    left = T.StructType([T.StructField("b", T.IntegerType(), True), T.StructField("a", T.StringType(), True)])
    right = T.StructType([T.StructField("a", T.StringType(), True), T.StructField("b", T.IntegerType(), True)])
    with pytest.raises(errors.PySparkAssertionError):
        assertSchemaEqual(left, right)
    assertSchemaEqual(left, right, ignoreColumnOrder=True)
    upper = T.StructType([T.StructField("X", T.IntegerType(), True)])
    lower = T.StructType([T.StructField("y", T.IntegerType(), True)])
    with pytest.raises(errors.PySparkAssertionError):
        assertSchemaEqual(upper, lower, ignoreColumnName=False)
    assertSchemaEqual(upper, lower, ignoreColumnName=True)
    with pytest.raises(errors.PySparkAssertionError):
        assertSchemaEqual(
            upper,
            T.StructType([T.StructField("y", T.IntegerType(), True), T.StructField("z", T.IntegerType(), True)]),
            ignoreColumnName=True,
        )
    with pytest.raises(errors.PySparkTypeError):
        assertSchemaEqual(left, "not a schema")
    df_wide = spark.createDataFrame([(1, 2)], ["a", "b"])
    df_narrow = spark.createDataFrame([(1,)], ["a"])
    with pytest.raises(errors.PySparkAssertionError):
        assertDataFrameEqual(df_wide, df_narrow, ignoreColumnType=True)
