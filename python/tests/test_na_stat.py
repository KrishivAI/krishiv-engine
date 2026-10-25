# SPDX-License-Identifier: Apache-2.0
"""``DataFrame.na`` and ``DataFrame.stat`` (``krishiv.sql.na`` and ``krishiv.sql.stat``).

Each test follows an example from the PySpark documentation of ``DataFrameNaFunctions``,
``DataFrameStatFunctions`` and the matching ``DataFrame`` methods, with the documented output.
Where the engine cannot run something yet (``sample`` and the variance, covariance, correlation,
percentile and frequent-items aggregates), the test asserts the engine's own
``KRISHIV_UNSUPPORTED_FEATURE`` error instead of emulating the result: those assertions are the ones
to turn into the documented outputs when the engine gains the feature.

String columns built from Python rows or pyarrow ``string`` columns are Arrow
``Utf8`` columns while string literals arrive as ``Utf8View``; the physical
planner meets the two carriers in the canonical one, so comparisons and
value-merging operators work across them.
"""

from __future__ import annotations

import contextlib
import doctest
import sys
import textwrap
from collections.abc import Generator
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, DataFrameNaFunctions, DataFrameStatFunctions, Row, SparkSession
from krishiv.sql import functions as F

pa = pytest.importorskip("pyarrow")

NAN = float("nan")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def arrow(values: list[Any], typ: Any = None) -> Any:
    """An Arrow array; strings are ``string_view`` unless ``typ`` says otherwise."""
    if typ is None and any(isinstance(v, str) for v in values):
        typ = pa.string_view()
    return pa.array(values, type=typ)


def frame(spark: SparkSession, columns: dict[str, list[Any]], **types: Any) -> DataFrame:
    return spark.createDataFrame(pa.table({name: arrow(values, types.get(name)) for name, values in columns.items()}))


def _order(value: Any) -> tuple[bool, bool, Any]:
    return (value is None, isinstance(value, str), 0 if value is None else value)


def rows(df: DataFrame) -> list[tuple[Any, ...]]:
    """The rows in a stable order (numbers, then strings, then NULL, per column), with NaN as the
    string ``"NaN"`` (it does not equal itself)."""
    out = [tuple("NaN" if isinstance(v, float) and v != v else v for v in r) for r in df.collect()]
    return sorted(out, key=lambda r: tuple(_order(v) for v in r))


def assert_shows(df: DataFrame, expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    df.show()
    assert capsys.readouterr().out == textwrap.dedent(expected).lstrip("\n")


@contextlib.contextmanager
def fails(cls: type[Exception], error_class: str, **parameters: str) -> Generator[None]:
    with pytest.raises(cls) as info:
        yield
    assert isinstance(info.value, errors.PySparkException)
    assert info.value.getCondition() == error_class
    for name, value in parameters.items():
        assert (info.value.getMessageParameters() or {})[name] == value


@contextlib.contextmanager
def unsupported(feature: str) -> Generator[None]:
    with fails(errors.UnsupportedOperationException, "KRISHIV_UNSUPPORTED_FEATURE", feature=feature):
        yield


@pytest.fixture
def dropna_people(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            Row(age=10, height=80.0, name="Alice"),
            Row(age=5, height=NAN, name="Bob"),
            Row(age=None, height=None, name="Tom"),
            Row(age=None, height=NAN, name=None),
        ]
    )


@pytest.fixture
def fill_people(spark: SparkSession) -> DataFrame:
    return frame(
        spark,
        {
            "age": [10, 5, None, None],
            "height": [80.5, None, None, None],
            "name": ["Alice", "Bob", "Tom", None],
            "bool": [None, None, None, True],
        },
    )


@pytest.fixture
def replace_people(spark: SparkSession) -> DataFrame:
    return frame(
        spark,
        {"age": [10, 5, None, None], "height": [80, None, 10, None], "name": ["Alice", "Bob", "Tom", None]},
    )


@pytest.fixture
def typed(spark: SparkSession) -> DataFrame:
    """One column of each numeric type, a string and a boolean, with nulls (and NaNs in the floats)."""
    return frame(
        spark,
        {
            "i": [1, None, 3],
            "t": [1, None, 3],
            "f": [1.5, NAN, None],
            "d": [1.5, None, 3.0],
            "s": ["a", None, "c"],
            "b": [True, None, False],
        },
        i=pa.int32(),
        t=pa.int8(),
        f=pa.float32(),
        d=pa.float64(),
    )


# DataFrame.na and DataFrame.stat ------------------------------------------------------------


def test_na_and_stat_are_properties_of_a_dataframe(typed: DataFrame) -> None:
    assert isinstance(typed.na, DataFrameNaFunctions)
    assert isinstance(typed.stat, DataFrameStatFunctions)
    assert typed.na.df is typed
    assert typed.stat.df is typed
    assert typed.columns == ["i", "t", "f", "d", "s", "b"]


def test_shortcuts_match_the_na_functions(typed: DataFrame) -> None:
    assert rows(typed.dropna(subset="s")) == rows(typed.na.drop(subset="s"))
    assert rows(typed.fillna(0, subset=["i"])) == rows(typed.na.fill(0, subset=["i"]))
    assert rows(typed.replace(1, 5, "i")) == rows(typed.na.replace(1, 5, "i"))
    assert rows(typed.replace({1: 5}, subset=["i"])) == rows(typed.na.replace({1: 5}, subset=["i"]))


# docstring examples as doctests --------------------------------------------------------------


def failing_examples(method: Any, spark: SparkSession) -> list[str]:
    """The sources of the examples in the docstring of ``method`` that do not give the documented output."""
    failed: list[str] = []

    class Runner(doctest.DocTestRunner):
        def report_failure(self, out: Any, test: Any, example: Any, got: str) -> None:
            failed.append(example.source.strip())

        def report_unexpected_exception(self, out: Any, test: Any, example: Any, exc_info: Any) -> None:
            failed.append(example.source.strip())

    test = doctest.DocTestParser().get_doctest(method.__doc__, {"spark": spark}, method.__name__, None, 0)
    Runner(verbose=False).run(test, out=lambda _: None)
    assert test.examples
    return failed


# The documented examples compare or coalesce a string column made from
# Python rows with a string literal; the engine runs them as written, so no
# example fails.
STRING_LITERAL_EXAMPLES: dict[str, list[str]] = {}


@pytest.mark.parametrize("name", ["dropna", "fillna", "replace", "crosstab"])
def test_docstring_examples_run_as_documented(spark: SparkSession, name: str) -> None:
    assert failing_examples(getattr(DataFrame, name), spark) == STRING_LITERAL_EXAMPLES.get(name, [])


# dropna / na.drop ---------------------------------------------------------------------------


def test_dropna_example_1_any(dropna_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        dropna_people.na.drop(),
        """
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        | 10|  80.0|Alice|
        +---+------+-----+
        """,
        capsys,
    )


def test_dropna_example_2_all(dropna_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        dropna_people.na.drop(how="all").sort("name"),
        """
        +----+------+-----+
        | age|height| name|
        +----+------+-----+
        |  10|  80.0|Alice|
        |   5|   NaN|  Bob|
        |NULL|  NULL|  Tom|
        +----+------+-----+
        """,
        capsys,
    )


def test_dropna_example_3_thresh(dropna_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        dropna_people.na.drop(thresh=2).sort("name"),
        """
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        | 10|  80.0|Alice|
        |  5|   NaN|  Bob|
        +---+------+-----+
        """,
        capsys,
    )


def test_dropna_example_4_subset(dropna_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        dropna_people.na.drop(subset=["age", "name"]).sort("name"),
        """
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        | 10|  80.0|Alice|
        |  5|   NaN|  Bob|
        +---+------+-----+
        """,
        capsys,
    )


def test_dropna_subset_forms_and_thresh_overrides_how(dropna_people: DataFrame) -> None:
    want = [(5, "NaN", "Bob"), (10, 80.0, "Alice")]
    for subset in ("age", ["age"], ("age",)):
        assert rows(dropna_people.dropna(subset=subset)) == want
    # ``thresh`` wins over ``how``: three non-null values are required, and only one row has them.
    assert rows(dropna_people.dropna(how="all", thresh=3)) == [(10, 80.0, "Alice")]
    assert rows(dropna_people.dropna(how="any", thresh=1)) == rows(dropna_people.dropna(how="all"))
    # ``how=None`` is treated as ``"all"``, as PySpark does.
    assert rows(dropna_people.dropna(how=None)) == rows(dropna_people.dropna(how="all"))


def test_dropna_counts_nan_as_missing_only_in_floating_columns(spark: SparkSession) -> None:
    df = frame(spark, {"x": [1.0, NAN, None, 4.0], "n": [1, 2, 3, None]}, n=pa.int64())
    assert rows(df.dropna(subset="x")) == [(1.0, 1), (4.0, None)]
    assert rows(df.dropna(subset="n")) == [(1.0, 1), ("NaN", 2), (None, 3)]
    assert rows(df.dropna()) == [(1.0, 1)]
    f32 = frame(spark, {"x": [1.5, NAN, None]}, x=pa.float32())
    assert rows(f32.dropna()) == [(1.5,)]


def test_dropna_thresh_bounds(dropna_people: DataFrame) -> None:
    assert dropna_people.dropna(thresh=0).count() == 4
    assert dropna_people.dropna(thresh=-1).count() == 4
    assert dropna_people.dropna(thresh=4).count() == 0
    assert dropna_people.dropna(thresh=3).count() == 1


def test_dropna_with_an_empty_subset(dropna_people: DataFrame) -> None:
    # Nothing is required of a row for "any", and one non-null value of none is impossible for "all".
    assert dropna_people.dropna(subset=[]).count() == 4
    assert dropna_people.dropna(how="all", subset=[]).count() == 0


def test_dropna_all_null_column_and_empty_dataframe(spark: SparkSession) -> None:
    df = frame(spark, {"a": [1, 2], "n": [None, None]}, n=pa.int64())
    assert df.dropna().count() == 0
    assert rows(df.dropna(subset="a")) == [(1, None), (2, None)]
    assert df.dropna(how="all").count() == 2
    empty = df.filter(F.lit(False))
    assert empty.dropna().dtypes == df.dtypes
    assert empty.dropna().count() == 0


def test_dropna_resolves_names_like_the_session(typed: DataFrame, spark: SparkSession) -> None:
    assert typed.dropna(subset="S").count() == 2
    assert typed.dropna(subset="`s`").count() == 2
    spark.conf.set("spark.sql.caseSensitive", "true")
    try:
        with pytest.raises(errors.AnalysisException):
            typed.dropna(subset="S")
    finally:
        spark.conf.set("spark.sql.caseSensitive", "false")


@pytest.mark.parametrize(
    ("kwargs", "condition", "parameters"),
    [
        ({"how": "some"}, "VALUE_NOT_ALLOWED", {"arg_name": "how", "allowed_values": "['any', 'all']"}),
        ({"how": 1}, "NOT_EXPECTED_TYPE", {"arg_name": "how", "expected_type": "str", "arg_type": "int"}),
        ({"thresh": "2"}, "NOT_EXPECTED_TYPE", {"arg_name": "thresh", "expected_type": "int", "arg_type": "str"}),
        (
            {"subset": 3},
            "NOT_EXPECTED_TYPE",
            {"arg_name": "subset", "expected_type": "list, str or tuple", "arg_type": "int"},
        ),
        ({"subset": ["s", 1]}, "NOT_EXPECTED_TYPE", {"arg_name": "subset", "expected_type": "str", "arg_type": "int"}),
    ],
)
def test_dropna_argument_errors(
    typed: DataFrame, kwargs: dict[str, Any], condition: str, parameters: dict[str, str]
) -> None:
    with fails(errors.PySparkTypeError if condition == "NOT_EXPECTED_TYPE" else errors.PySparkValueError, condition):
        typed.dropna(**kwargs)
    with fails(errors.PySparkException, condition, **parameters):
        typed.na.drop(**kwargs)


def test_dropna_names_that_are_not_columns(typed: DataFrame) -> None:
    with pytest.raises(errors.AnalysisException) as info:
        typed.dropna(subset=["s", "nope"])
    assert info.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    with unsupported("DataFrame.na on the qualified or nested column reference q.s"):
        typed.alias("q").dropna(subset="q.s")


# fillna / na.fill ---------------------------------------------------------------------------


def test_fillna_example_1_numbers(fill_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        fill_people.na.fill(50),
        """
        +---+------+-----+----+
        |age|height| name|bool|
        +---+------+-----+----+
        | 10|  80.5|Alice|NULL|
        |  5|  50.0|  Bob|NULL|
        | 50|  50.0|  Tom|NULL|
        | 50|  50.0| NULL|true|
        +---+------+-----+----+
        """,
        capsys,
    )


def test_fillna_example_2_boolean(fill_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        fill_people.na.fill(False),
        """
        +----+------+-----+-----+
        | age|height| name| bool|
        +----+------+-----+-----+
        |  10|  80.5|Alice|false|
        |   5|  NULL|  Bob|false|
        |NULL|  NULL|  Tom|false|
        |NULL|  NULL| NULL| true|
        +----+------+-----+-----+
        """,
        capsys,
    )


def test_fillna_example_3_dict(fill_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        fill_people.na.fill({"age": 50, "name": "unknown"}),
        """
        +---+------+-------+----+
        |age|height|   name|bool|
        +---+------+-------+----+
        | 10|  80.5|  Alice|NULL|
        |  5|  NULL|    Bob|NULL|
        | 50|  NULL|    Tom|NULL|
        | 50|  NULL|unknown|true|
        +---+------+-------+----+
        """,
        capsys,
    )


def test_fillna_example_4_string_subset(fill_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        fill_people.na.fill(value="Spark", subset="name"),
        """
        +----+------+-----+----+
        | age|height| name|bool|
        +----+------+-----+----+
        |  10|  80.5|Alice|NULL|
        |   5|  NULL|  Bob|NULL|
        |NULL|  NULL|  Tom|NULL|
        |NULL|  NULL|Spark|true|
        +----+------+-----+----+
        """,
        capsys,
    )


def test_fillna_keeps_column_types_and_order(typed: DataFrame) -> None:
    for value in (7, 7.5, "x", True):
        assert typed.fillna(value).dtypes == typed.dtypes
        assert typed.fillna(value).columns == typed.columns
    filled = typed.fillna(7)
    assert rows(filled) == [
        (1, 1, 1.5, 1.5, "a", True),
        (3, 3, 7.0, 3.0, "c", False),
        (7, 7, 7.0, 7.0, None, None),
    ]


def test_fillna_casts_the_number_to_the_column_type(typed: DataFrame) -> None:
    # 3.7 into an integer column is 3, as the cast to the column type (and not a type change).
    assert rows(typed.fillna(3.7).select("i", "t", "d")) == [(1, 1, 1.5), (3, 3, 3.0), (3, 3, 3.7)]
    assert typed.fillna(3.7).select("f").dtypes == [("f", "float")]


def test_fillna_replaces_nan_in_floating_columns_only(spark: SparkSession) -> None:
    df = frame(spark, {"f": [1.5, NAN, None], "d": [NAN, 2.0, None], "n": [1, None, 3]}, f=pa.float32(), n=pa.int64())
    assert rows(df.fillna(0)) == [(0.0, 0.0, 3), (0.0, 2.0, 0), (1.5, 0.0, 1)]
    assert df.fillna(0).dtypes == df.dtypes


def test_fillna_value_types_pick_the_columns(typed: DataFrame) -> None:
    # A value of another type than a column's leaves that column alone.
    assert rows(typed.fillna("x").select("i", "d", "s", "b")) == [
        (1, 1.5, "a", True),
        (3, 3.0, "c", False),
        (None, None, "x", None),
    ]
    assert rows(typed.fillna(True).select("i", "s", "b")) == [(1, "a", True), (3, "c", False), (None, None, True)]
    assert rows(typed.fillna(1).select("s", "b")) == [("a", True), ("c", False), (None, None)]


def test_fillna_type_mismatched_subset_is_ignored(typed: DataFrame) -> None:
    assert rows(typed.fillna("x", subset=["i", "d"])) == rows(typed)
    assert rows(typed.fillna(0, subset="s")) == rows(typed)
    assert rows(typed.fillna(False, subset=["i", "f"])) == rows(typed)
    mixed = typed.fillna("x", subset=["i", "s"])
    assert [r for r in rows(mixed) if r[0] is None] == [(None, None, "NaN", None, "x", None)]


def test_fillna_subset_forms(typed: DataFrame) -> None:
    for subset in ("i", ["i"], ("i",), "I", "`i`"):
        out = typed.fillna(0, subset=subset)
        assert rows(out.select("i", "t")) == [(0, None), (1, 1), (3, 3)]
    assert rows(typed.fillna(0, subset=["i", "t"]).select("i", "t")) == [(0, 0), (1, 1), (3, 3)]
    assert rows(typed.fillna(0, subset=[]).select("i")) == rows(typed.select("i"))


def test_fillna_dict_casts_each_value_to_its_column(typed: DataFrame) -> None:
    out = typed.fillna({"i": 5, "f": 2, "s": "q", "t": True, "d": 0.5})
    assert out.dtypes == typed.dtypes
    assert rows(out) == [(1, 1, 1.5, 1.5, "a", True), (3, 3, 2.0, 3.0, "c", False), (5, 1, 2.0, 0.5, "q", None)]
    assert rows(typed.fillna({"I": 5}, subset="s")) == rows(typed.fillna({"i": 5}))
    assert rows(typed.fillna({})) == rows(typed)


def test_fillna_dict_value_the_column_cannot_hold_fails_at_the_action(typed: DataFrame) -> None:
    out = typed.fillna({"i": "x"})
    with pytest.raises(errors.NumberFormatException) as info:
        out.collect()
    assert info.value.getCondition() == "CAST_INVALID_INPUT"
    with pytest.raises(errors.ArithmeticException) as overflow:
        typed.fillna(300).collect()
    assert overflow.value.getCondition() == "CAST_OVERFLOW"


def test_fillna_all_null_columns_and_empty_dataframes(spark: SparkSession) -> None:
    df = frame(spark, {"n": [None, None], "s": [None, None]}, n=pa.int64(), s=pa.string_view())
    assert rows(df.fillna(1)) == [(1, None), (1, None)]
    assert rows(df.fillna({"s": "z"})) == [(None, "z"), (None, "z")]
    empty = df.filter(F.lit(False))
    assert empty.fillna(1).dtypes == df.dtypes
    assert empty.fillna(1).count() == 0


def test_fillna_takes_integers_as_doubles(spark: SparkSession) -> None:
    # As PySpark (``classic/dataframe.py``: ``value = float(value)`` for non-bool ints), so one
    # beyond 64 bits still fills a double column ...
    df = frame(spark, {"d": [None, 1.0]})
    assert rows(df.fillna(2**70)) == [(1.0,), (2.0**70,)]
    # ... and a non-representable int on an integer column rounds the way Spark's
    # double-to-long cast does.
    big = frame(spark, {"n": [None, 1]}, n=pa.int64())
    assert rows(big.fillna(2**62 + 1)) == [(1,), (2**62,)]


def test_names_with_backquotes(spark: SparkSession) -> None:
    df = frame(spark, {"we`ird": [None, 1]}, **{"we`ird": pa.int64()})
    assert rows(df.fillna(0, subset="`we``ird`")) == [(0,), (1,)]
    with pytest.raises(errors.AnalysisException):
        df.fillna(0, subset="`we`ird`")


def test_fillna_decimal_column(spark: SparkSession) -> None:
    df = spark.createDataFrame(pa.table({"x": pa.array([None, 1], pa.decimal128(10, 2))}))
    out = df.fillna(2.5)
    assert out.dtypes == [("x", "decimal(10,2)")]
    assert sorted(str(r[0]) for r in out.collect()) == ["1.00", "2.50"]


@pytest.mark.parametrize(
    ("args", "arg_name", "expected", "arg_type"),
    [
        ((None,), "value", "bool, dict, float, int or str", "NoneType"),
        (([1],), "value", "bool, dict, float, int or str", "list"),
        ((1, 5), "subset", "list or tuple", "int"),
        ((1, ["i", 2]), "subset", "str", "int"),
        (({1: 2},), "key type of dict", "str", "int"),
        (({"i": None},), "value type of dict", "bool, float, int or str", "NoneType"),
        (({"i": [1]},), "value type of dict", "bool, float, int or str", "list"),
    ],
)
def test_fillna_argument_errors(
    typed: DataFrame, args: tuple[Any, ...], arg_name: str, expected: str, arg_type: str
) -> None:
    with fails(
        errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name=arg_name, expected_type=expected, arg_type=arg_type
    ):
        typed.fillna(*args)


def test_fillna_names_that_are_not_columns(typed: DataFrame) -> None:
    for call in (
        lambda: typed.fillna(1, subset="nope"),
        lambda: typed.fillna({"nope": 1}),
        lambda: typed.fillna({"i": 1, "nope": 1}),
    ):
        with pytest.raises(errors.AnalysisException) as info:
            call()
        assert info.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    with unsupported("DataFrame.na on the qualified or nested column reference q.i"):
        typed.alias("q").fillna(1, subset="q.i")
    with unsupported("DataFrame.na on the qualified or nested column reference q.i"):
        typed.alias("q").fillna({"q.i": 1})


def test_na_functions_on_a_join_with_duplicate_column_names(spark: SparkSession) -> None:
    left = frame(spark, {"k": [1, 2, 3], "v": [1.5, None, 3.0], "s": ["a", None, "c"]}, k=pa.int64())
    right = frame(spark, {"k": [1, 2, 3], "v": [None, 20.0, NAN]}, k=pa.int64())
    joined = left.join(right, left["k"] == right["k"])
    assert joined.columns == ["k", "v", "s", "k", "v"]
    filled = joined.fillna(0)
    assert filled.columns == joined.columns
    assert filled.dtypes == joined.dtypes
    assert rows(filled) == [(1, 1.5, "a", 1, 0.0), (2, 0.0, None, 2, 20.0), (3, 3.0, "c", 3, 0.0)]
    assert rows(joined.fillna("x")) == [(1, 1.5, "a", 1, None), (2, None, "x", 2, 20.0), (3, 3.0, "c", 3, "NaN")]
    assert joined.dropna().columns == joined.columns
    assert rows(joined.dropna()) == []  # every row has a null or NaN in one of the two ``v``
    assert rows(joined.dropna(thresh=4)) == [(1, 1.5, "a", 1, None), (3, 3.0, "c", 3, "NaN")]
    assert rows(joined.replace(1, 5)) == [(2, None, None, 2, 20.0), (3, 3.0, "c", 3, "NaN"), (5, 1.5, "a", 5, None)]
    for call in (
        lambda: joined.fillna(0, subset="v"),
        lambda: joined.dropna(subset="k"),
        lambda: joined.replace(1, 2, "v"),
    ):
        with pytest.raises(errors.AnalysisException) as info:
            call()
        assert info.value.getCondition() == "AMBIGUOUS_REFERENCE"


def test_has_duplicates_follows_the_session_case_sensitivity() -> None:
    from krishiv.sql.dataframe import _has_duplicates

    assert _has_duplicates(["A", "a"], False)
    assert not _has_duplicates(["A", "a"], True)
    assert not _has_duplicates(["A", "b"], False)
    assert _has_duplicates(["k", "v", "k"], True)


# replace / na.replace -----------------------------------------------------------------------


def test_replace_example_1_number_in_all_columns(replace_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        replace_people.na.replace(10, 20),
        """
        +----+------+-----+
        | age|height| name|
        +----+------+-----+
        |  20|    80|Alice|
        |   5|  NULL|  Bob|
        |NULL|    20|  Tom|
        |NULL|  NULL| NULL|
        +----+------+-----+
        """,
        capsys,
    )


def test_replace_example_2_string_with_null(replace_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        replace_people.na.replace("Alice", None),
        """
        +----+------+----+
        | age|height|name|
        +----+------+----+
        |  10|    80|NULL|
        |   5|  NULL| Bob|
        |NULL|    10| Tom|
        |NULL|  NULL|NULL|
        +----+------+----+
        """,
        capsys,
    )


def test_replace_example_3_lists_in_a_column(replace_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        replace_people.na.replace(["Alice", "Bob"], ["A", "B"], "name"),
        """
        +----+------+----+
        | age|height|name|
        +----+------+----+
        |  10|    80|   A|
        |   5|  NULL|   B|
        |NULL|    10| Tom|
        |NULL|  NULL|NULL|
        +----+------+----+
        """,
        capsys,
    )


def test_replace_example_4_number_in_a_column(replace_people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert_shows(
        replace_people.na.replace(10, 18, "age"),
        """
        +----+------+-----+
        | age|height| name|
        +----+------+-----+
        |  18|    80|Alice|
        |   5|  NULL|  Bob|
        |NULL|    10|  Tom|
        |NULL|  NULL| NULL|
        +----+------+-----+
        """,
        capsys,
    )


def test_replace_with_a_dict(typed: DataFrame) -> None:
    out = typed.replace({1: 10, 3: 30}, subset=["i", "t"])
    assert rows(out.select("i", "t", "d")) == [(10, 10, 1.5), (30, 30, 3.0), (None, None, None)]
    # The value is not needed with a dict, and is ignored (with a warning) when given.
    with pytest.warns(UserWarning, match="value will be ignored"):
        ignored = typed.replace({1: 10}, 99, subset="i")
    assert rows(ignored.select("i")) == rows(typed.replace({1: 10}, subset="i").select("i"))
    assert rows(typed.replace({"a": "z", "c": None}).select("s")) == [("z",), (None,), (None,)]


def test_replace_with_sequences(typed: DataFrame) -> None:
    by_list = typed.replace([1, 3], [100, 300], subset=["i"])
    assert rows(by_list.select("i")) == [(100,), (300,), (None,)]
    assert rows(typed.replace((1, 3), (100, 300), subset=("i",)).select("i")) == rows(by_list.select("i"))
    # One value for every item of the list.
    assert rows(typed.replace([1, 3], 0, subset="i").select("i")) == [(0,), (0,), (None,)]
    # A scalar `to_replace` with a longer `value` list keeps the first pair, as PySpark's
    # `dict(zip(...))` does (lengths are only checked when both are sequences).
    assert rows(typed.replace(1, [100, 300], subset="i").select("i")) == rows(
        typed.replace(1, 100, subset="i").select("i")
    )
    assert rows(typed.replace(["a", "c"], None).select("s")) == [(None,), (None,), (None,)]
    assert rows(typed.replace([], [], subset="i")) == rows(typed)


def test_replace_keeps_large_integers_exact(spark: SparkSession) -> None:
    # 2**62 + 1 is not exactly representable as a double: a float() detour would round the
    # key to 2**62 and replace both rows, where Spark (ints stay ints) replaces one.
    df = frame(spark, {"x": [2**62 + 1, 2**62, None]}, x=pa.int64())
    assert rows(df.replace(2**62 + 1, 0)) == [(0,), (2**62,), (None,)]
    assert rows(df.replace(2**62, 2**62 + 1)) == [(2**62 + 1,), (2**62 + 1,), (None,)]


def test_replace_casts_the_new_value_to_the_column_type(typed: DataFrame) -> None:
    # 2.9 into an integer column is 2: the cast to the column's type, which the replacement keeps.
    out = typed.replace(1, 2.9)
    assert out.dtypes == typed.dtypes
    assert rows(out.select("i", "t")) == [(2, 2), (3, 3), (None, None)]
    # A key matches by value across the numeric types: 1.5 is in the float and the double column.
    assert rows(typed.replace(1.5, 3.5).select("f", "d", "i")) == [(3.5, 3.5, 1), ("NaN", None, None), (None, 3.0, 3)]
    # NaN equals NaN here, as in Spark.
    assert rows(typed.replace(NAN, 0.0).select("f")) == [(0.0,), (1.5,), (None,)]


def test_replace_with_none_for_a_number(typed: DataFrame) -> None:
    out = typed.replace(3, None)
    assert out.dtypes == typed.dtypes
    assert rows(out.select("i", "t", "d")) == [(1, 1, 1.5), (None, None, None), (None, None, None)]


def test_replace_value_types_pick_the_columns(typed: DataFrame) -> None:
    assert rows(typed.replace("a", "b")) == [
        (1, 1, 1.5, 1.5, "b", True),
        (3, 3, None, 3.0, "c", False),
        (None, None, "NaN", None, None, None),
    ]
    assert rows(typed.replace(1, 7).select("s", "b")) == rows(typed.select("s", "b"))
    assert rows(typed.replace(True, False).select("i", "b")) == [(1, False), (3, False), (None, None)]
    assert rows(typed.replace(False, None, "b").select("b")) == [(True,), (None,), (None,)]


def test_replace_with_nothing_to_do(typed: DataFrame) -> None:
    assert rows(typed.replace({})) == rows(typed)
    assert rows(typed.replace(1, 2, subset=[])) == rows(typed.replace(1, 2, subset="s"))
    assert rows(typed.replace(99, 2)) == rows(typed)
    all_null = typed.filter(F.col("i").isNull()).select("t")
    assert rows(all_null.replace(1, 2)) == [(None,)]
    empty = typed.filter(F.lit(False))
    assert empty.replace(1, 2).dtypes == typed.dtypes
    assert empty.replace(1, 2).count() == 0


@pytest.mark.parametrize(
    ("args", "error", "condition", "parameters"),
    [
        (
            (1,),
            errors.PySparkTypeError,
            "ARGUMENT_REQUIRED",
            {"arg_name": "value", "condition": "`to_replace` is dict"},
        ),
        (
            (None, 1),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "to_replace", "expected_type": "bool, dict, float, int, str or tuple", "arg_type": "NoneType"},
        ),
        (
            (1, object()),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "value", "expected_type": "bool, float, int, list, None, str or tuple", "arg_type": "object"},
        ),
        (
            ([1, 2], [1]),
            errors.PySparkValueError,
            "LENGTH_SHOULD_BE_THE_SAME",
            {"arg1": "to_replace", "arg2": "value", "arg1_length": "2", "arg2_length": "1"},
        ),
        (
            (1, 2, 3),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "subset", "expected_type": "list, str or tuple", "arg_type": "int"},
        ),
        ((1, 2, ["i", 1]), errors.PySparkTypeError, "NOT_EXPECTED_TYPE", {"arg_name": "subset", "arg_type": "int"}),
        ((1, "a"), errors.PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
        (("a", 1), errors.PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
        ((1, True), errors.PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
        (({1: 2, "a": "b"},), errors.PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
        (([1, "a"], 0), errors.PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
        (({1: "a"},), errors.PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
        (([None], 1), errors.PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
    ],
)
def test_replace_argument_errors(
    typed: DataFrame, args: tuple[Any, ...], error: type[Exception], condition: str, parameters: dict[str, str]
) -> None:
    with fails(error, condition, **parameters):
        typed.replace(*args)


def test_replace_names_that_are_not_columns(typed: DataFrame) -> None:
    with pytest.raises(errors.AnalysisException) as info:
        typed.replace(1, 2, subset=["i", "nope"])
    assert info.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    with unsupported("DataFrame.na on the qualified or nested column reference q.i"):
        typed.alias("q").replace(1, 2, subset="q.i")


# crosstab -----------------------------------------------------------------------------------


def test_crosstab_example(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = spark.createDataFrame([(1, 11), (1, 11), (3, 10), (4, 8), (4, 8)], ["c1", "c2"])
    assert_shows(
        df.crosstab("c1", "c2").sort("c1_c2"),
        """
        +-----+---+---+---+
        |c1_c2| 10| 11|  8|
        +-----+---+---+---+
        |    1|  0|  2|  0|
        |    3|  1|  0|  0|
        |    4|  0|  0|  2|
        +-----+---+---+---+
        """,
        capsys,
    )
    assert df.stat.crosstab("c1", "c2").dtypes == [
        ("c1_c2", "string"),
        ("10", "bigint"),
        ("11", "bigint"),
        ("8", "bigint"),
    ]


def test_crosstab_null_keys_and_labels_are_named_null(spark: SparkSession) -> None:
    df = frame(spark, {"a": ["x", "x", None, "y"], "b": ["p", None, "p", "p"]})
    out = df.stat.crosstab("a", "b")
    assert out.columns == ["a_b", "null", "p"]
    assert rows(out) == [("null", 0, 1), ("x", 1, 1), ("y", 0, 1)]


def test_crosstab_labels_drop_backquotes_and_sort_as_strings(spark: SparkSession) -> None:
    df = frame(spark, {"a": [1, 1, 2], "b": ["z`z", "B", "10"]})
    out = df.crosstab("a", "b")
    assert out.columns == ["a_b", "10", "B", "zz"]
    assert rows(out) == [("1", 0, 1, 1), ("2", 1, 0, 0)]


def test_crosstab_of_numbers_names_the_columns_by_their_text(spark: SparkSession) -> None:
    df = frame(spark, {"a": [1.5, 1.5, NAN], "b": [True, False, True]})
    out = df.crosstab("a", "b")
    assert out.columns == ["a_b", "false", "true"]
    assert rows(out) == [("1.5", 1, 1), ("NaN", 0, 1)]


def test_crosstab_of_an_empty_dataframe_has_only_the_key_column(spark: SparkSession) -> None:
    df = frame(spark, {"a": [1], "b": [2]}).filter(F.lit(False))
    out = df.crosstab("a", "b")
    assert out.dtypes == [("a_b", "string")]
    assert out.count() == 0


def test_crosstab_with_a_label_named_like_the_key_column_keeps_both(spark: SparkSession) -> None:
    # The first column is ``a_b`` and so is the column of the value ``a_b``: neither is dropped or renamed.
    # Spark's own behavior here is unverified (no spark-golden coverage); this pins the facade's choice.
    df = frame(spark, {"a": ["u", "v"], "b": ["a_b", "w"]})
    out = df.crosstab("a", "b")
    assert out.columns == ["a_b", "a_b", "w"]
    assert rows(out) == [("u", 1, 0), ("v", 0, 1)]


def test_crosstab_limits_the_distinct_values_of_the_second_column(spark: SparkSession) -> None:
    df = frame(spark, {"a": [1, 1, 1], "b": [1, 2, 3]})
    spark.conf.set("spark.sql.pivotMaxValues", "3")
    try:
        assert df.crosstab("a", "b").columns == ["a_b", "1", "2", "3"]
        spark.conf.set("spark.sql.pivotMaxValues", "2")
        with pytest.raises(errors.AnalysisException, match="more than 2 distinct values") as info:
            df.crosstab("a", "b")
        assert info.value.getCondition() == "_LEGACY_ERROR_TEMP_1324"
        spark.conf.set("spark.sql.pivotMaxValues", "not a number")
        assert df.crosstab("a", "b").columns == ["a_b", "1", "2", "3"]
        spark.conf.set("spark.sql.pivotMaxValues", "99999999999999999999")  # beyond a 32-bit integer
        assert df.crosstab("a", "b").columns == ["a_b", "1", "2", "3"]
        spark.conf.set("spark.sql.pivotMaxValues", "-5")  # nothing is allowed
        with pytest.raises(errors.AnalysisException, match="more than 0 distinct values"):
            df.crosstab("a", "b")
    finally:
        spark.conf.unset("spark.sql.pivotMaxValues")


def test_crosstab_argument_errors(spark: SparkSession) -> None:
    df = frame(spark, {"a": [1], "b": [2]})
    with fails(errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name="col1", expected_type="str", arg_type="int"):
        df.crosstab(1, "b")  # ty: ignore[invalid-argument-type]
    with fails(errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name="col2", expected_type="str", arg_type="NoneType"):
        df.stat.crosstab("a", None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.AnalysisException) as info:
        df.crosstab("a", "nope")
    assert info.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"


# sampleBy -----------------------------------------------------------------------------------


@pytest.fixture
def sampled_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, float, int]]:
    """``DataFrame.sample`` (the engine cannot sample yet) replaced by a recorder that keeps every row."""
    calls: list[tuple[bool, float, int]] = []

    def sample(self: DataFrame, withReplacement: bool, fraction: float, seed: int) -> DataFrame:
        calls.append((withReplacement, fraction, seed))
        return self

    monkeypatch.setattr(DataFrame, "sample", sample)
    return calls


def test_sample_by_samples_each_stratum_and_unions_them(
    spark: SparkSession, sampled_calls: list[tuple[bool, float, int]]
) -> None:
    df = frame(spark, {"k": [0, 0, 1, 2, None, 1], "v": ["a", "b", "c", "d", "e", "f"]})
    out = df.sampleBy("k", {0: 0.1, 1: 0.5}, seed=11)
    assert out.columns == ["k", "v"]
    assert rows(out) == [(0, "a"), (0, "b"), (1, "c"), (1, "f")]  # stratum 2 and the null key are in no sample
    assert sampled_calls == [(False, 0.1, 11), (False, 0.5, 11)]


def test_sample_by_is_reproducible_with_a_seed(
    spark: SparkSession, sampled_calls: list[tuple[bool, float, int]]
) -> None:
    df = frame(spark, {"k": ["x", "y"]})
    first = df.stat.sampleBy(F.col("k"), {"x": 1.0, "y": 0.25}, 3)
    again = df.stat.sampleBy("k", {"x": 1.0, "y": 0.25}, 3)
    assert rows(first) == rows(again)
    assert sampled_calls == [(False, 1.0, 3), (False, 0.25, 3)] * 2
    sampled_calls.clear()
    df.sampleBy("k", {"x": 0.5, "y": 0.5})
    (seed,) = {call[2] for call in sampled_calls}  # without a seed one is drawn, and shared by the strata
    assert 0 <= seed <= sys.maxsize


def test_sample_by_keeps_the_fractions_it_was_given(
    spark: SparkSession, sampled_calls: list[tuple[bool, float, int]]
) -> None:
    fractions = {1: 1, "a": 0}
    frame(spark, {"k": [1, 2]}).sampleBy("k", fractions, 1)  # ty: ignore[invalid-argument-type]
    assert fractions == {1: 1, "a": 0}
    assert sampled_calls[0] == (False, 1.0, 1)


def test_sample_by_without_fractions_is_empty(spark: SparkSession) -> None:
    df = frame(spark, {"k": [1, 2], "v": ["a", "b"]})
    out = df.sampleBy("k", {})
    assert out.dtypes == df.dtypes
    assert out.count() == 0


@pytest.mark.parametrize(
    ("args", "error", "condition", "parameters"),
    [
        (
            (1, {0: 0.5}),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "col", "expected_type": "Column or str"},
        ),
        (
            ("k", [0.5]),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "fractions", "expected_type": "dict"},
        ),
        (
            ("k", {None: 0.5}),
            errors.PySparkTypeError,
            "DISALLOWED_TYPE_FOR_CONTAINER",
            {"arg_name": "fractions", "allowed_types": "float, int, str", "item_type": "NoneType"},
        ),
    ],
)
def test_sample_by_argument_errors(
    spark: SparkSession, args: tuple[Any, ...], error: type[Exception], condition: str, parameters: dict[str, str]
) -> None:
    with fails(error, condition, **parameters):
        frame(spark, {"k": [1]}).sampleBy(*args)


@pytest.mark.parametrize("fraction", [1.5, -0.1, NAN])
def test_sample_by_fractions_must_be_in_the_unit_interval(spark: SparkSession, fraction: float) -> None:
    with pytest.raises(errors.IllegalArgumentException, match=r"Fractions must be in \[0, 1\]"):
        frame(spark, {"k": [1]}).sampleBy("k", {1: 0.5, 2: fraction})
    with pytest.raises(ValueError, match="could not convert"):
        frame(spark, {"k": [1]}).sampleBy("k", {1: "x"})  # ty: ignore[invalid-argument-type]


def test_sample_by_example_fails_with_the_engines_sample_error(spark: SparkSession) -> None:
    # The documented outputs (4 and 9 rows, then 33) need the engine's ``sample``.
    dataset = spark.range(0, 100, 1, 5).select((F.col("id") % 3).alias("key"))
    sampled = dataset.sampleBy("key", fractions={0: 0.1, 1: 0.2}, seed=0)  # lazy: the error is the action's
    with unsupported("DataFrame.sample"):
        sampled.groupBy("key").count().orderBy("key").show()
    with unsupported("DataFrame.sample"):
        dataset.sampleBy(F.col("key"), fractions={2: 1.0}, seed=0).count()


# describe, summary, corr, cov, freqItems, approxQuantile -------------------------------------


@pytest.fixture
def measures(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [("Bob", 13, 40.3, 150.5), ("Alice", 12, 37.8, 142.3), ("Tom", 11, 44.1, 142.2)],
        ["name", "age", "weight", "height"],
    )


def test_describe_examples_fail_with_the_engines_error(measures: DataFrame) -> None:
    with unsupported("DataFrame.describe"):
        measures.describe(["age"])
    with unsupported("DataFrame.describe"):
        measures.describe(["age", "weight", "height"])
    with unsupported("DataFrame.describe"):
        measures.describe("age", "weight")
    with unsupported("DataFrame.describe"):
        measures.describe()


def test_summary_examples_fail_with_the_engines_error(measures: DataFrame) -> None:
    numbers = measures.select("age", "weight", "height")
    with unsupported("DataFrame.summary"):
        numbers.summary()
    with unsupported("DataFrame.summary"):
        numbers.summary("count", "min", "25%", "75%", "max")
    with unsupported("DataFrame.summary"):
        numbers.summary(["count", "max"])


def test_describe_and_summary_argument_errors(measures: DataFrame) -> None:
    with fails(errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name="cols", expected_type="str", arg_type="int"):
        measures.describe("age", 1)  # ty: ignore[invalid-argument-type]
    with fails(errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name="statistics", arg_type="int"):
        measures.summary("count", 1)  # ty: ignore[invalid-argument-type]
    with fails(errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name="statistics", arg_type="int"):
        measures.summary([1])  # ty: ignore[invalid-argument-type]


def test_corr_and_cov_examples_fail_with_the_engines_error(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, 12), (10, 1), (19, 8)], ["c1", "c2"])
    bigger = spark.createDataFrame([(11, 12), (10, 11), (9, 10)], ["small", "bigger"])
    for call, feature in [
        (lambda: df.corr("c1", "c2"), "DataFrame.corr"),
        (lambda: bigger.corr("small", "bigger"), "DataFrame.corr"),
        (lambda: df.stat.corr("c1", "c2", "pearson"), "DataFrame.corr"),
        (lambda: df.cov("c1", "c2"), "DataFrame.cov"),
        (lambda: bigger.stat.cov("small", "bigger"), "DataFrame.cov"),
    ]:
        with unsupported(feature):
            call()


def test_corr_and_cov_argument_errors(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, 12)], ["c1", "c2"])
    with fails(errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name="col1", expected_type="str", arg_type="int"):
        df.corr(1, "c2")  # ty: ignore[invalid-argument-type]
    with fails(errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name="col2", expected_type="str", arg_type="list"):
        df.cov("c1", ["c2"])  # ty: ignore[invalid-argument-type]
    with fails(errors.PySparkValueError, "VALUE_NOT_PEARSON", arg_name="method", arg_value="spearman"):
        df.corr("c1", "c2", "spearman")
    # A falsy method is the default "pearson" (as PySpark's `if not method`), so "" reaches the engine.
    with unsupported("DataFrame.corr"):
        df.corr("c1", "c2", "")


def test_freq_items_example_fails_with_the_engines_error(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, 11), (1, 11), (3, 10), (4, 8), (4, 8)], ["c1", "c2"])
    with unsupported("DataFrame.freqItems"):
        df.freqItems(["c1", "c2"])
    with unsupported("DataFrame.freqItems"):
        df.stat.freqItems(("c1",), 0.5)
    with fails(
        errors.PySparkTypeError, "NOT_EXPECTED_TYPE", arg_name="cols", expected_type="list or tuple", arg_type="str"
    ):
        df.freqItems("c1")  # ty: ignore[invalid-argument-type]


def test_freq_items_bad_support_and_columns_reach_the_engines_error(spark: SparkSession) -> None:
    # PySpark checks only the `cols` container client-side and hands the rest to the engine
    # (`if not support: support = 0.01`), so with no sketch every call past the check fails
    # with the engine's own error.
    df = spark.createDataFrame([(1, 11), (1, 11)], ["c1", "c2"])
    with unsupported("DataFrame.freqItems"):
        df.freqItems(["c1"], "x")  # ty: ignore[invalid-argument-type]
    with unsupported("DataFrame.freqItems"):
        df.freqItems(["c1"], -1)
    with unsupported("DataFrame.freqItems"):
        df.freqItems(["c1", 1])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("data", "names", "column", "probabilities", "error"),
    [
        ([(1,), (2,), (3,), (4,), (5,)], ["values"], "values", [0.0, 0.5, 1.0], 0.05),
        ([(1, 10), (2, 20), (3, 30)], ["col1", "col2"], ["col1", "col2"], [0.0, 0.5, 1.0], 0.05),
        ([(1,), (None,), (3,), (4,), (None,)], ["values"], "values", [0.0, 0.5, 1.0], 0.05),
        ([(1,), (2,), (3,), (4,), (5,)], ["values"], "values", [0.0, 0.2, 1.0], 0.1),
    ],
)
def test_approx_quantile_examples_fail_with_the_engines_error(
    spark: SparkSession,
    data: list[tuple[Any, ...]],
    names: list[str],
    column: Any,
    probabilities: list[float],
    error: float,
) -> None:
    df = spark.createDataFrame(data, names)
    with unsupported("DataFrame.approxQuantile"):
        df.approxQuantile(column, probabilities, error)
    with unsupported("DataFrame.approxQuantile"):
        df.stat.approxQuantile(tuple(column) if isinstance(column, list) else column, tuple(probabilities), error)


@pytest.mark.parametrize(
    ("args", "error", "condition", "parameters"),
    [
        ((1, [0.5], 0.1), errors.PySparkTypeError, "NOT_EXPECTED_TYPE", {"arg_name": "col", "arg_type": "int"}),
        (
            (["v", 1], [0.5], 0.1),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "columns", "arg_type": "int"},
        ),
        (
            ("v", 0.5, 0.1),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "probabilities", "arg_type": "float"},
        ),
        (("v", [0.5, 1.5], 0.1), errors.PySparkTypeError, "NOT_EXPECTED_TYPE", {"arg_name": "probabilities"}),
        (("v", [-0.5], 0.1), errors.PySparkTypeError, "NOT_EXPECTED_TYPE", {"arg_name": "probabilities"}),
        (
            ("v", ["x"], 0.1),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "probabilities", "arg_type": "str"},
        ),
        (
            ("v", [0.5], "0.1"),
            errors.PySparkTypeError,
            "NOT_EXPECTED_TYPE",
            {"arg_name": "relativeError", "arg_type": "str"},
        ),
        (
            ("v", [0.5], -1),
            errors.PySparkValueError,
            "NEGATIVE_VALUE",
            {"arg_name": "relativeError", "arg_value": "-1"},
        ),
    ],
)
def test_approx_quantile_argument_errors(
    spark: SparkSession, args: tuple[Any, ...], error: type[Exception], condition: str, parameters: dict[str, str]
) -> None:
    with fails(error, condition, **parameters):
        spark.createDataFrame([(1,)], ["v"]).approxQuantile(*args)


def test_error_messages_are_pysparks(typed: DataFrame) -> None:
    calls = [
        (lambda: typed.replace(1), "[ARGUMENT_REQUIRED] Argument `value` is required when `to_replace` is dict."),
        (lambda: typed.replace(1, "a"), "[MIXED_TYPE_REPLACEMENT] Mixed type replacements are not supported."),
        (
            lambda: typed.replace([1, 2], [1]),
            "[LENGTH_SHOULD_BE_THE_SAME] to_replace and value should be of the same length, got 2 and 1.",
        ),
        (
            lambda: typed.approxQuantile("i", [0.5], -1),
            "[NEGATIVE_VALUE] Value for `relativeError` must be greater than or equal to 0, got '-1'.",
        ),
        (
            lambda: typed.corr("i", "t", "spearman"),
            "[VALUE_NOT_PEARSON] Value for `method` only supports 'pearson', got 'spearman'.",
        ),
        (
            lambda: typed.dropna(how="some"),
            "[VALUE_NOT_ALLOWED] Value for `how` has to be amongst the following values: ['any', 'all'].",
        ),
        (
            lambda: typed.fillna(None),  # ty: ignore[invalid-argument-type]
            "[NOT_EXPECTED_TYPE] Argument `value` should be bool, dict, float, int or str, got NoneType.",
        ),
    ]
    for call, message in calls:
        with pytest.raises(errors.PySparkException) as info:
            call()
        assert str(info.value) == message


def test_unsupported_errors_carry_the_sql_state(spark: SparkSession) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as info:
        spark.createDataFrame([(1, 2)], ["a", "b"]).cov("a", "b")
    assert info.value.getSqlState() == "0A000"
    assert "[KRISHIV_UNSUPPORTED_FEATURE]" in str(info.value)
