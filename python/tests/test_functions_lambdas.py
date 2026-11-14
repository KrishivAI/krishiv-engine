# SPDX-License-Identifier: Apache-2.0
"""The higher-order, generator, sort-order and unsupported ``pyspark.sql.functions`` names (PY-016) and
``Column.__pow__`` (PY-041). Expected values are PySpark 4.2.0's, from the oracle replay of the sweep."""

from __future__ import annotations

from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, SparkSession
from krishiv.sql import functions as F
from krishiv.sql.column import Column

pa = pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def df() -> DataFrame:
    spark = SparkSession.builder.create()
    return spark.sql(
        "select array(3,1,2) arr, array(2,3,4) arr2, map('a',1,'b',2) m, "
        "array(named_struct('k','a','v',1)) astr, 5 i, 7 l, 2.5 n"
    )


def one(df: DataFrame, column: Any) -> Any:
    return df.select(column.alias("r")).collect()[0][0]


# Higher-order functions.


def test_transform_with_and_without_index(df: DataFrame) -> None:
    assert one(df, F.transform("arr", lambda x: x + 1)) == [4, 2, 3]
    assert one(df, F.transform("arr", lambda x, i: x * i)) == [0, 1, 4]


def test_filter_exists_forall(df: DataFrame) -> None:
    assert one(df, F.filter("arr", lambda x: x > 1)) == [3, 2]
    assert one(df, F.exists("arr", lambda x: x > 2)) is True
    assert one(df, F.forall("arr", lambda x: x > 2)) is False


def test_aggregate_and_reduce(df: DataFrame) -> None:
    assert one(df, F.aggregate("arr", F.lit(0), lambda acc, x: acc + x)) == 6
    assert one(df, F.aggregate("arr", F.lit(0), lambda acc, x: acc + x, lambda acc: acc * 10)) == 60
    assert one(df, F.reduce("arr", F.lit(1), lambda acc, x: acc * x)) == 6


def test_zip_with_and_array_sort(df: DataFrame) -> None:
    assert one(df, F.zip_with("arr", "arr2", lambda a, b: a + b)) == [5, 4, 6]
    assert one(df, F.array_sort("arr")) == [1, 2, 3]
    assert one(df, F.array_sort("arr", lambda a, b: F.when(a < b, 1).when(a > b, -1).otherwise(0))) == [3, 2, 1]


def test_map_functions(df: DataFrame) -> None:
    assert one(df, F.map_filter("m", lambda k, v: v > 1)) == {"b": 2}
    assert one(df, F.transform_values("m", lambda k, v: v + 10)) == {"a": 11, "b": 12}
    assert one(df, F.transform_keys("m", lambda k, v: F.upper(k))) == {"A": 1, "B": 2}
    assert one(df, F.map_zip_with("m", "m", lambda k, a, b: a + b)) == {"a": 2, "b": 4}


def test_lambda_variables_do_not_capture_columns() -> None:
    """A column named like a lambda variable of an earlier version (``x_0``) is read as the column."""
    spark = SparkSession.builder.create()
    d = spark.sql("select array(1,2) arr, 100 x_0, 1000 y_0, 10000 x_1")
    assert one(d, F.transform("arr", lambda x: x + F.col("x_0"))) == [101, 102]
    assert one(d, F.transform("arr", lambda x: x + F.col("x_1"))) == [10001, 10002]
    assert one(d, F.aggregate("arr", F.lit(0), lambda acc, x: acc + x + F.col("y_0"))) == 2003


@pytest.mark.xfail(reason="the engine's filter takes no index lambda: INVALID_LAMBDA_FUNCTION_CALL", strict=True)
def test_filter_with_an_index_lambda(df: DataFrame) -> None:
    assert one(df, F.filter("arr", lambda x, i: i > 0)) == [1, 2]


def test_nested_lambdas_do_not_capture(df: DataFrame) -> None:
    got = one(df, F.transform("arr", lambda x: F.aggregate("arr2", F.lit(0), lambda acc, y: acc + y * x)))
    assert got == [27, 9, 18]


def test_lambda_argument_checks(df: DataFrame) -> None:
    with pytest.raises(errors.PySparkValueError, match="WRONG_NUM_ARGS_FOR_HIGHER_ORDER_FUNCTION"):
        F.transform("arr", lambda: F.lit(1))
    with pytest.raises(errors.PySparkValueError, match="WRONG_NUM_ARGS_FOR_HIGHER_ORDER_FUNCTION"):
        F.transform("arr", lambda x, y, z, w: x)
    # PySpark accepts one to three parameters when the function is built; a wrong count for this higher-order
    # function fails when the query is analyzed.
    bad = F.exists("arr", lambda x, y: x)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.AnalysisException, match="NUM_ARGS_MISMATCH"):
        df.select(bad).collect()
    with pytest.raises(errors.PySparkValueError, match="HIGHER_ORDER_FUNCTION_SHOULD_RETURN_COLUMN"):
        F.transform("arr", lambda x: 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkValueError, match="UNSUPPORTED_PARAM_TYPE_FOR_HIGHER_ORDER_FUNCTION"):
        F.transform("arr", lambda *xs: xs[0])
    with pytest.raises(errors.PySparkTypeError):
        F.transform("arr", 1)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


# Generators build the call that SQL `explode(col)` is; running them through select is the DataFrame
# generator work (PR-C), so these wait for it.

GENERATORS = ["explode", "explode_outer", "posexplode", "posexplode_outer", "inline", "inline_outer"]


@pytest.mark.parametrize("name", GENERATORS)
def test_generators_build_their_call(name: str) -> None:
    column = getattr(F, name)("arr")
    assert isinstance(column, Column)


@pytest.mark.xfail(reason="generators in select() wait for the DataFrame generator work (PR-C)", strict=True)
def test_explode_in_select(df: DataFrame) -> None:
    assert [r[0] for r in df.select(F.explode("arr")).collect()] == [3, 1, 2]


@pytest.mark.xfail(reason="generators in select() wait for the DataFrame generator work (PR-C)", strict=True)
def test_posexplode_and_inline_in_select(df: DataFrame) -> None:
    assert [tuple(r) for r in df.select(F.posexplode("arr")).collect()] == [(0, 3), (1, 1), (2, 2)]
    assert [tuple(r) for r in df.select(F.inline("astr")).collect()] == [("a", 1)]


# Sort orders, call_udf, broadcast.


def test_sort_orders_with_nulls() -> None:
    spark = SparkSession.builder.create()
    d = spark.sql("select * from values (1), (null), (2) as t(a)")
    assert [r[0] for r in d.orderBy(F.asc_nulls_first("a")).collect()] == [None, 1, 2]
    assert [r[0] for r in d.orderBy(F.asc_nulls_last("a")).collect()] == [1, 2, None]
    assert [r[0] for r in d.orderBy(F.desc_nulls_first("a")).collect()] == [None, 2, 1]
    assert [r[0] for r in d.orderBy(F.desc_nulls_last("a")).collect()] == [2, 1, None]


def test_call_udf_calls_a_registered_function(df: DataFrame) -> None:
    assert one(df, F.call_udf("abs", F.lit(-3))) == 3


def test_broadcast_returns_the_hinted_dataframe(df: DataFrame) -> None:
    assert isinstance(F.broadcast(df), DataFrame)


# The newly wrapped registered functions.


def test_power_and_aliases(df: DataFrame) -> None:
    assert one(df, F.power("i", 2)) == 25.0
    assert one(df, F.negate("i")) == -5
    assert one(df, F.chr(F.lit(65))) == "A"


def test_column_pow_and_rpow(df: DataFrame) -> None:
    assert one(df, F.col("i") ** 2) == 25.0
    assert one(df, 2 ** F.col("i")) == 32.0
    assert one(df, F.col("i") ** F.col("i")) == 3125.0


def test_aggregates_added(df: DataFrame) -> None:
    assert df.select(F.max_by("i", "l")).collect()[0][0] == 5
    assert df.select(F.min_by("i", "l")).collect()[0][0] == 5
    assert df.select(F.any_value("i")).collect()[0][0] == 5
    assert df.select(F.array_agg("i")).collect()[0][0] == [5]
    row = df.select(F.every(F.lit(True)), F.some(F.lit(False))).collect()[0]
    assert (row[0], row[1]) == (True, False)


# What the engine lacks: a clear error, never an AttributeError.

UNSUPPORTED = [
    ("product", ("n",)),
    ("sum_distinct", ("n",)),
    ("sumDistinct", ("n",)),
    ("udtf", ()),
    ("unwrap_udt", ("n",)),
    ("window", ("n", "10 minutes")),
    ("current_time", ()),
    ("st_srid", ("n",)),
]


@pytest.mark.parametrize(("name", "args"), UNSUPPORTED)
def test_missing_engine_functions_raise_not_implemented(name: str, args: tuple[Any, ...]) -> None:
    with pytest.raises(NotImplementedError):
        getattr(F, name)(*args)


def test_catalog_only_functions_raise_unsupported_when_called() -> None:
    with pytest.raises(errors.UnsupportedOperationException, match="KRISHIV_UNSUPPORTED_FEATURE"):
        F.bitmap_count("n")


def test_udtf_stubs_take_pysparks_keywords() -> None:
    with pytest.raises(NotImplementedError):
        F.udtf(returnType="a int")
    with pytest.raises(NotImplementedError):
        F.arrow_udtf(returnType="a int")


def test_random_is_rand_by_another_name(df: DataFrame) -> None:
    assert one(df, F.random(1)) == one(df, F.rand(1))
    assert isinstance(one(df, F.random()), float)


def _unsupported_names() -> list[str]:
    import inspect

    out = []
    for name, fn in vars(F).items():
        if inspect.isfunction(fn) and fn.__module__ == F.__name__:
            try:
                src = inspect.getsource(fn)
            except OSError:
                continue
            if "raise not_implemented(" in src:
                out.append(name)
    return sorted(out)


@pytest.mark.parametrize("name", _unsupported_names())
def test_every_unsupported_name_raises_not_implemented(name: str) -> None:
    import inspect

    sig = inspect.signature(getattr(F, name))
    args = [
        "n"
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty and p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    with pytest.raises(NotImplementedError):
        getattr(F, name)(*args)


def test_stack_builds_its_call() -> None:
    assert isinstance(F.stack(F.lit(2), F.lit(1), F.lit(2)), Column)
