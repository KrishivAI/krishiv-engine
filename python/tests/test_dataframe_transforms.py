# SPDX-License-Identifier: Apache-2.0
"""PySpark-shaped ``DataFrame`` transforms (P2-T186 part A): distinct, dropDuplicates, drop,
renames, set operations, sample, repartition/coalesce, offset, tail, hint, unpivot/melt,
transform, toDF, toLocalIterator and toJSON.

Each test follows a PySpark 4.2 ``DataFrame`` docstring example, with the expected output from
the PySpark docs. Where the engine cannot run something yet, the test asserts the engine's
own error instead of emulating it.
"""

from __future__ import annotations

from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, Row, SparkSession
from krishiv.sql import functions as F

pa = pytest.importorskip("pyarrow")


def make(spark: SparkSession, data: Any) -> DataFrame:
    """A DataFrame of the docstring rows, via Arrow (``createDataFrame`` takes Arrow data only)."""
    if isinstance(data, list):
        return spark.createDataFrame(pa.Table.from_pylist(data))
    if isinstance(data, dict):
        return spark.createDataFrame(pa.table(data))
    return spark.createDataFrame(data)


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def people(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        pa.table({"age": [14, 23, 16], "name": ["Tom", "Alice", "Bob"]}),
    )


def dicts(df: DataFrame, *sort: str) -> list[dict[str, Any]]:
    rows = [r.asDict() for r in df.collect()]
    if sort:
        rows.sort(key=lambda r: [(r[c] is None, str(r[c])) for c in sort])
    return rows


def shown(df: DataFrame, capsys: pytest.CaptureFixture[str], *args: Any, **kwargs: Any) -> str:
    df.show(*args, **kwargs)
    return capsys.readouterr().out


# sort / orderBy ----------------------------------------------------------------------------------


def test_sort_with_ascending_lists(people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert shown(people.sort("age", ascending=[False]), capsys) == (
        "+---+-----+\n|age| name|\n+---+-----+\n| 23|Alice|\n| 16|  Bob|\n| 14|  Tom|\n+---+-----+\n"
    )
    assert [r["age"] for r in people.orderBy("name", "age", ascending=[True, False]).collect()] == [23, 16, 14]
    assert [r["age"] for r in people.sort(F.col("age").desc()).collect()] == [23, 16, 14]
    with pytest.raises(errors.PySparkValueError):
        people.sort("age", "name", ascending=[True])


# distinct ---------------------------------------------------------------------------------


def test_distinct(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = make(spark, {"age": [14, 23, 23], "name": ["Tom", "Alice", "Alice"]})
    assert shown(df.distinct().sort("age"), capsys) == (
        "+---+-----+\n|age| name|\n+---+-----+\n| 14|  Tom|\n| 23|Alice|\n+---+-----+\n"
    )
    assert df.distinct().count() == 2
    assert df.select("name").distinct().count() == 2
    # NULL equals NULL, as in SQL.
    nulled = make(spark, {"age": [14, 14, 23], "name": [None, None, "x"]})
    assert dicts(nulled.distinct(), "age") == [
        {"age": 14, "name": None},
        {"age": 23, "name": "x"},
    ]


def test_set_operations_reject_non_dataframes(spark: SparkSession) -> None:
    df = make(spark, {"a": [1]})
    with pytest.raises(errors.PySparkTypeError):
        df.union("nope")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.unionAll("nope")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.unionByName("nope")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.intersect("nope")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.intersectAll("nope")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.exceptAll("nope")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.subtract("nope")  # ty: ignore[invalid-argument-type]


def test_distinct_rejects_non_dataframes(spark: SparkSession) -> None:
    df = make(spark, {"a": [1]})
    with pytest.raises(errors.PySparkTypeError):
        df.union("nope")  # ty: ignore[invalid-argument-type]


# dropDuplicates -----------------------------------------------------------------------------


def test_drop_duplicates(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = make(
        spark,
        [
            {"name": "Alice", "age": 5, "height": 80},
            {"name": "Alice", "age": 5, "height": 80},
            {"name": "Alice", "age": 10, "height": 80},
        ],
    )
    assert shown(df.dropDuplicates().sort("age"), capsys) == (
        "+-----+---+------+\n"
        "| name|age|height|\n"
        "+-----+---+------+\n"
        "|Alice|  5|    80|\n"
        "|Alice| 10|    80|\n"
        "+-----+---+------+\n"
    )
    assert shown(df.dropDuplicates(["name", "height"]).sort("age"), capsys) == (
        "+-----+---+------+\n| name|age|height|\n+-----+---+------+\n|Alice|  5|    80|\n+-----+---+------+\n"
    )
    assert df.drop_duplicates(["name", "height"]).count() == 1
    assert df.drop_duplicates(("name", "height")).count() == 1


def test_drop_duplicates_subset_must_be_str_list(spark: SparkSession) -> None:
    df = make(spark, {"a": [1]})
    with pytest.raises(errors.PySparkTypeError) as bad:
        df.dropDuplicates("a")  # ty: ignore[invalid-argument-type]
    assert bad.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert bad.value.getMessageParameters() == {
        "arg_name": "subset",
        "expected_type": "list or tuple",
        "arg_type": "str",
    }
    with pytest.raises(errors.PySparkTypeError):
        df.dropDuplicates([1])  # ty: ignore[invalid-argument-type]


def test_drop_duplicates_unknown_column_raises_the_engine_error(spark: SparkSession) -> None:
    df = make(spark, {"a": [1]})
    with pytest.raises(errors.AnalysisException) as bad:
        df.dropDuplicates(["nope"]).collect()
    assert (bad.value.getCondition() or "").startswith("UNRESOLVED_COLUMN")


# drop ---------------------------------------------------------------------------------------


def test_drop(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = make(spark, {"age": [14, 23, 16], "name": ["Tom", "Alice", "Bob"]})
    assert shown(df.drop("age"), capsys) == ("+-----+\n| name|\n+-----+\n|  Tom|\n|Alice|\n|  Bob|\n+-----+\n")
    assert shown(df.drop(df.age), capsys) == ("+-----+\n| name|\n+-----+\n|  Tom|\n|Alice|\n|  Bob|\n+-----+\n")
    # A missing column is a no-op.
    assert df.drop("height").columns == ["age", "name"]
    assert df.drop("height", "weight").columns == ["age", "name"]
    assert df.drop().columns == ["age", "name"]


def test_drop_two_columns_with_the_same_name(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = make(spark, {"age": [14, 23, 16], "name": ["Tom", "Alice", "Bob"]})
    df2 = make(spark, {"height": [80, 85], "name": ["Tom", "Bob"]})
    joined = df.join(df2, df.name == df2.name).drop("name").sort("age")
    assert joined.columns == ["age", "height"]
    assert shown(joined, capsys) == (
        "+---+------+\n|age|height|\n+---+------+\n| 14|    80|\n| 16|    85|\n+---+------+\n"
    )
    both = df.join(df2).drop("name")
    assert both.columns == ["age", "height"]


def test_drop_column_with_dots_in_its_name(spark: SparkSession) -> None:
    df = make(spark, {"age": [14], "name": ["Tom"]}).withColumn("a.b.c", F.lit(1))
    assert df.drop("a.b.c").columns == ["age", "name"]
    # By Column it resolves as a nested field, which matches nothing: a no-op.
    assert df.drop(F.col("a.b.c")).columns == ["age", "name", "a.b.c"]


def test_drop_ambiguous_column_raises(spark: SparkSession) -> None:
    df = make(spark, {"age": [14], "name": ["Tom"]})
    df2 = make(spark, {"height": [80], "name": ["Tom"]})
    with pytest.raises(errors.AnalysisException) as bad:
        _ = df.join(df2).drop(F.col("name")).columns
    assert bad.value.getCondition() == "AMBIGUOUS_REFERENCE"


def test_drop_rejects_other_types(spark: SparkSession) -> None:
    df = make(spark, {"a": [1]})
    with pytest.raises(errors.PySparkTypeError):
        df.drop(1)  # ty: ignore[invalid-argument-type]


# withColumnRenamed / withColumnsRenamed --------------------------------------------------------


def test_with_column_renamed(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = make(spark, {"age": [2, 5], "name": ["Alice", "Bob"]})
    assert shown(df.withColumnRenamed("age", "age2"), capsys) == (
        "+----+-----+\n|age2| name|\n+----+-----+\n|   2|Alice|\n|   5|  Bob|\n+----+-----+\n"
    )
    assert df.withColumnRenamed("non_existing", "new_name").columns == ["age", "name"]
    assert df.withColumnRenamed("age", "age2").withColumnRenamed("name", "name2").columns == [
        "age2",
        "name2",
    ]
    with pytest.raises(errors.PySparkTypeError):
        df.withColumnRenamed(1, "x")  # ty: ignore[invalid-argument-type]


def test_with_columns_renamed(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = make(spark, {"age": [2, 5], "name": ["Alice", "Bob"]})
    assert shown(df.withColumnsRenamed({"age": "age2"}), capsys) == (
        "+----+-----+\n|age2| name|\n+----+-----+\n|   2|Alice|\n|   5|  Bob|\n+----+-----+\n"
    )
    renamed = df.withColumnsRenamed({"age": "age2", "name": "name2"})
    assert renamed.columns == ["age2", "name2"]
    assert dicts(renamed, "age2") == [
        {"age2": 2, "name2": "Alice"},
        {"age2": 5, "name2": "Bob"},
    ]
    assert df.withColumnsRenamed({"non_existing": "new_name"}).columns == ["age", "name"]
    assert df.withColumnsRenamed({}).columns == ["age", "name"]
    with pytest.raises(errors.PySparkTypeError):
        df.withColumnsRenamed(["age"])  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.withColumnsRenamed({"age": 1})  # ty: ignore[invalid-argument-type]


# union ---------------------------------------------------------------------------------------


def test_union(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df1 = make(spark, {"id": [1, 2], "value": ["A", "B"]})
    df2 = make(spark, {"id": [3, 4], "value": ["C", "D"]})
    assert shown(df1.union(df2).sort("id"), capsys) == (
        "+---+-----+\n| id|value|\n+---+-----+\n|  1|    A|\n|  2|    B|\n|  3|    C|\n|  4|    D|\n+---+-----+\n"
    )
    # Columns resolve by position, not by name.
    left = make(spark, {"A": [1], "B": [2]})
    right = make(spark, {"C": [3], "D": [4]})
    assert dicts(left.union(right), "A") == [{"A": 1, "B": 2}, {"A": 3, "B": 4}]
    # No deduplication: union then distinct.
    duped = make(spark, {"id": [1, 2, 3], "value": ["A", "B", "C"]}).union(
        make(spark, {"id": [3, 4], "value": ["C", "D"]})
    )
    assert dicts(duped.distinct().sort("id"), "id") == [
        {"id": 1, "value": "A"},
        {"id": 2, "value": "B"},
        {"id": 3, "value": "C"},
        {"id": 4, "value": "D"},
    ]
    # Types widen per position, as in SQL.
    assert df1.union(df1.select(F.col("id").cast("long"), "value")).dtypes == [
        ("id", "bigint"),
        ("value", "string"),
    ]


def test_union_all_is_union(spark: SparkSession) -> None:
    df = make(spark, {"a": [1]})
    assert df.unionAll(df).count() == 2


def test_union_rejects_mismatched_widths(spark: SparkSession) -> None:
    df = make(spark, {"a": [1], "b": [2]})
    with pytest.raises(errors.AnalysisException) as bad:
        df.union(make(spark, {"a": [1]})).collect()
    assert bad.value.getCondition() == "NUM_COLUMNS_MISMATCH"


def test_union_by_name(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df1 = make(spark, {"col0": [1], "col1": [2], "col2": [3]})
    df2 = make(spark, {"col1": [4], "col2": [5], "col0": [6]})
    assert shown(df1.unionByName(df2).sort("col0"), capsys) == (
        "+----+----+----+\n|col0|col1|col2|\n+----+----+----+\n|   1|   2|   3|\n|   6|   4|   5|\n+----+----+----+\n"
    )


def test_union_by_name_with_missing_columns(spark: SparkSession) -> None:
    df1 = make(spark, {"col0": [1], "col1": [2], "col2": [3]})
    df2 = make(spark, {"col1": [4], "col2": [5], "col3": [6]})
    assert dicts(df1.unionByName(df2, allowMissingColumns=True), "col1", "col2") == [
        {"col0": 1, "col1": 2, "col2": 3, "col3": None},
        {"col0": None, "col1": 4, "col2": 5, "col3": 6},
    ]
    df3 = make(spark, {"col0": [0], "col1": [1], "col2": [2]})
    df4 = make(spark, {"col1": [3], "col2": [4], "col3": [5], "col4": [6]})
    assert dicts(df3.unionByName(df4, allowMissingColumns=True), "col1", "col2") == [
        {"col0": 0, "col1": 1, "col2": 2, "col3": None, "col4": None},
        {"col0": None, "col1": 3, "col2": 4, "col3": 5, "col4": 6},
    ]
    df5 = make(spark, {"col3": [3], "col4": [4], "col5": [5]})
    assert dicts(df3.unionByName(df5, allowMissingColumns=True), "col0") == [
        {"col0": 0, "col1": 1, "col2": 2, "col3": None, "col4": None, "col5": None},
        {"col0": None, "col1": None, "col2": None, "col3": 3, "col4": 4, "col5": 5},
    ]


def test_union_by_name_with_missing_string_columns_raises_the_engine_error(
    spark: SparkSession,
) -> None:
    # The engine plans the null-filled branch but its physical planner rejects the union of a
    # computed NULL string with a scanned one (`Utf8` vs `Utf8View`); the same fails in SQL.
    left = make(spark, {"a": ["x"], "b": [1]})
    right = make(spark, {"b": [2], "c": ["y"]})
    with pytest.raises(errors.SparkRuntimeException) as bad:
        left.unionByName(right, allowMissingColumns=True).collect()
    assert bad.value.getCondition() == "INTERNAL_ERROR"


def test_union_by_name_rejects_missing_without_allow_missing(spark: SparkSession) -> None:
    df1 = make(spark, {"col0": [1], "col1": [2], "col2": [3]})
    df2 = make(spark, {"col1": [4], "col2": [5]})
    with pytest.raises(errors.AnalysisException) as bad:
        df1.unionByName(df2).collect()
    assert (bad.value.getCondition() or "").startswith("UNRESOLVED_COLUMN")


# intersect / intersectAll / exceptAll / subtract -------------------------------------------------


def test_intersect(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df1 = make(spark, {"C1": ["a", "a", "b", "c"], "C2": [1, 1, 3, 4]})
    df2 = make(spark, {"C1": ["a", "a", "b"], "C2": [1, 1, 3]})
    assert shown(df1.intersect(df2).sort("C1", "C2"), capsys) == (
        "+---+---+\n| C1| C2|\n+---+---+\n|  a|  1|\n|  b|  3|\n+---+---+\n"
    )
    other = make(spark, {"id": [2], "value": ["B"]})
    assert dicts(make(spark, {"id": [1, 2], "value": ["A", "B"]}).intersect(other), "id") == [{"id": 2, "value": "B"}]


def test_intersect_all(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df1 = make(spark, {"C1": ["a", "a", "b", "c"], "C2": [1, 1, 3, 4]})
    df2 = make(spark, {"C1": ["a", "a", "b"], "C2": [1, 1, 3]})
    assert shown(df1.intersectAll(df2).sort("C1", "C2"), capsys) == (
        "+---+---+\n| C1| C2|\n+---+---+\n|  a|  1|\n|  a|  1|\n|  b|  3|\n+---+---+\n"
    )


def test_subtract(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df1 = make(spark, {"C1": ["a", "a", "b", "c"], "C2": [1, 1, 3, 4]})
    df2 = make(spark, {"C1": ["a", "a", "b"], "C2": [1, 1, 3]})
    assert shown(df1.subtract(df2), capsys) == ("+---+---+\n| C1| C2|\n+---+---+\n|  c|  4|\n+---+---+\n")
    assert dicts(
        make(spark, {"id": [1, 2], "value": ["A", "B"]}).subtract(make(spark, {"id": [2, 3], "value": ["B", "C"]}))
    ) == [{"id": 1, "value": "A"}]


def test_except_all(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df1 = make(
        spark,
        {"C1": ["a", "a", "a", "a", "b", "c"], "C2": [1, 1, 1, 2, 3, 4]},
    )
    df2 = make(spark, {"C1": ["a", "b"], "C2": [1, 3]})
    assert shown(df1.exceptAll(df2).sort("C1", "C2"), capsys) == (
        "+---+---+\n| C1| C2|\n+---+---+\n|  a|  1|\n|  a|  1|\n|  a|  2|\n|  c|  4|\n+---+---+\n"
    )


# sample ----------------------------------------------------------------------------------------


def test_sample_is_unsupported(spark: SparkSession) -> None:
    df = make(spark, {"age": [14], "name": ["Tom"]})
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.sample(0.5, 3).count()
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert bad.value.getSqlState() == "0A000"
    with pytest.raises(errors.UnsupportedOperationException):
        df.sample(fraction=0.5, seed=3).count()
    with pytest.raises(errors.UnsupportedOperationException):
        df.sample(True, 0.5, 3).count()
    with pytest.raises(errors.UnsupportedOperationException):
        df.sample(withReplacement=False, fraction=0.5).count()


def test_sample_rejects_bad_arguments(spark: SparkSession) -> None:
    df = make(spark, {"age": [14], "name": ["Tom"]})
    with pytest.raises(errors.PySparkTypeError) as bad:
        df.sample()
    assert bad.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert bad.value.getMessageParameters() == {
        "expected_type": "bool, float or int",
        "arg_name": "withReplacement (optional), fraction (required) and seed (optional)",
        "arg_type": "NoneType, NoneType, NoneType",
    }
    with pytest.raises(errors.PySparkTypeError):
        df.sample(1)


# repartition / coalesce --------------------------------------------------------------------------


def test_repartition_and_coalesce_keep_the_rows(people: DataFrame) -> None:
    assert dicts(people.repartition(10).sort("age"), "age") == dicts(people.sort("age"), "age")
    assert dicts(people.repartition(7, "age").sort("age"), "age") == dicts(people.sort("age"), "age")
    assert dicts(people.repartition("age", "name").sort("age"), "age") == dicts(people.sort("age"), "age")
    assert people.coalesce(1).count() == 3
    for bad in (lambda: people.repartition(0), lambda: people.repartition(-1, "age"), lambda: people.coalesce(0)):
        with pytest.raises(
            errors.PySparkValueError, match=r"\[VALUE_NOT_POSITIVE\] Value for `numPartitions` must be positive"
        ):
            bad()
    with pytest.raises(errors.PySparkTypeError):
        people.repartition(1, 1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        people.repartition(1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        people.coalesce("x")  # ty: ignore[invalid-argument-type]


# offset / tail ------------------------------------------------------------------------------------


def test_offset(people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert shown(people.sort("age").offset(1), capsys) == (
        "+---+-----+\n|age| name|\n+---+-----+\n| 16|  Bob|\n| 23|Alice|\n+---+-----+\n"
    )
    assert people.sort("age").offset(10).collect() == []
    with pytest.raises(errors.PySparkTypeError):
        people.offset("1")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.AnalysisException):
        people.offset(-1).collect()


def test_tail(people: DataFrame) -> None:
    assert people.sort("age").tail(2) == [Row(age=16, name="Bob"), Row(age=23, name="Alice")]
    assert people.sort("age").tail(99) == people.sort("age").collect()
    assert people.sort("age").tail(0) == []
    with pytest.raises(errors.AnalysisException, match=r"INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE.*tail"):
        people.tail(-1)
    with pytest.raises(errors.PySparkTypeError):
        people.tail("2")  # ty: ignore[invalid-argument-type]


# hint -----------------------------------------------------------------------------------------------


def test_hint_is_accepted_and_recorded(people: DataFrame) -> None:
    assert people.hint("BROADCAST").count() == 3
    assert people.hint("broadcast", "people").count() == 3
    assert people.hint("broadcast", ["people"]).count() == 3
    assert people.hint("COALESCE", 1).count() == 3
    assert people.hint("BROADCAST", F.col("name")).count() == 3
    assert people.hint("BROADCAST", "age", ["name"]).count() == 3
    assert "BROADCAST" in people.hint("BROADCAST").explainString(mode="extended")
    with pytest.raises(errors.PySparkTypeError):
        people.hint(1, "x")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError) as bad:
        people.hint("BROADCAST", {"a": 1})
    assert bad.value.getCondition() == "DISALLOWED_TYPE_FOR_CONTAINER"
    with pytest.raises(errors.PySparkTypeError) as bad:
        people.hint("BROADCAST", "x", [1, {}])
    assert bad.value.getCondition() == "DISALLOWED_TYPE_FOR_CONTAINER"
    params = bad.value.getMessageParameters()
    assert isinstance(params, dict)
    assert params["item_type"] == "list[int]"


# unpivot / melt ---------------------------------------------------------------------------------------


def test_unpivot(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = make(spark, {"id": [1, 2], "int": [11, 12], "double": [1.1, 1.2]})
    assert shown(df.unpivot("id", ["int", "double"], "var", "val").sort("id", "var"), capsys) == (
        "+---+------+----+\n| id|   var| val|\n+---+------+----+\n"
        "|  1|double| 1.1|\n|  1|   int|11.0|\n"
        "|  2|double| 1.2|\n|  2|   int|12.0|\n+---+------+----+\n"
    )


def test_unpivot_defaults_to_all_non_id_columns(spark: SparkSession) -> None:
    df = make(spark, {"id": [1, 2], "int": [11, 12], "double": [1.1, 1.2]})
    assert dicts(df.unpivot("id", None, "var", "val").sort("id", "var"), "id", "var") == [
        {"id": 1, "var": "double", "val": 1.1},
        {"id": 1, "var": "int", "val": 11.0},
        {"id": 2, "var": "double", "val": 1.2},
        {"id": 2, "var": "int", "val": 12.0},
    ]
    # melt is an alias, and NULLs are kept, as in Spark's DataFrame API.
    nulled = make(spark, pa.table({"id": [1], "x": pa.array([None], pa.int64())}))
    assert dicts(nulled.melt("id", ["x"], "var", "val")) == [{"id": 1, "var": "x", "val": None}]
    # Columns work as ids and values too.
    assert dicts(df.unpivot(F.col("id"), [F.col("int")], "var", "val"), "id") == [
        {"id": 1, "var": "int", "val": 11.0},
        {"id": 2, "var": "int", "val": 12.0},
    ]
    assert dicts(df.unpivot(F.col("id"), None, "var", "val").sort("id", "var"), "id", "var") == [
        {"id": 1, "var": "double", "val": 1.1},
        {"id": 1, "var": "int", "val": 11.0},
        {"id": 2, "var": "double", "val": 1.2},
        {"id": 2, "var": "int", "val": 12.0},
    ]


def test_unpivot_rejects_bad_arguments(spark: SparkSession) -> None:
    df = make(spark, {"id": [1], "int": [11]})
    with pytest.raises(AssertionError):
        df.unpivot(None, ["int"], "var", "val")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.unpivot("id", [1], "var", "val")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.unpivot(1, ["int"], "var", "val")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.unpivot("id", ["int"], 1, "val")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.unpivot("id", ["int"], "var", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.unpivot("id", [], "var", "val").collect()
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"


# transform / toDF / toLocalIterator / toJSON --------------------------------------------------------------


def test_transform(people: DataFrame) -> None:
    def cast_all_to_int(input_df: DataFrame) -> DataFrame:
        return input_df.select([F.col(c).cast("int") for c in input_df.columns])

    assert people.transform(cast_all_to_int).dtypes == [("age", "int"), ("name", "int")]

    def add_n(input_df: DataFrame, n: int) -> DataFrame:
        return input_df.select((F.col("age") + n).alias("age"), "name")

    got = [r.asDict() for r in people.transform(add_n, 1).transform(add_n, n=10).sort("age").collect()]
    assert got == [
        {"age": 25, "name": "Tom"},
        {"age": 27, "name": "Bob"},
        {"age": 34, "name": "Alice"},
    ]
    with pytest.raises(AssertionError):
        people.transform(lambda df: 1)


def test_to_df(people: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert shown(people.sort("age").toDF("f1", "f2"), capsys) == (
        "+---+-----+\n| f1|   f2|\n+---+-----+\n| 14|  Tom|\n| 16|  Bob|\n| 23|Alice|\n+---+-----+\n"
    )
    with pytest.raises(errors.PySparkTypeError):
        people.toDF("a", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.AnalysisException) as bad:
        people.toDF("only").collect()
    assert bad.value.getCondition() == "ASSIGNMENT_ARITY_MISMATCH"


def test_to_local_iterator(people: DataFrame) -> None:
    assert list(people.sort("age").toLocalIterator()) == [
        Row(age=14, name="Tom"),
        Row(age=16, name="Bob"),
        Row(age=23, name="Alice"),
    ]


def test_to_json_is_unsupported(people: DataFrame) -> None:
    with pytest.raises(errors.UnsupportedOperationException, match="KRISHIV_UNSUPPORTED_FEATURE"):
        people.toJSON()


def test_drop_duplicates_on_every_column(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, "a"), (1, "a"), (2, "b")], ["k", "v"])
    assert df.dropDuplicates(["k", "v"]).sort("k").collect() == [Row(k=1, v="a"), Row(k=2, v="b")]
    one = spark.createDataFrame([(1,), (1,)], ["a"])
    assert one.dropDuplicates(["a"]).collect() == [Row(a=1)]
    with pytest.raises(errors.AnalysisException):
        _ = one.dropDuplicates(["missing"]).columns


def test_drop_and_rename_keep_duplicate_join_columns(spark: SparkSession) -> None:
    # Spark 4.2 keeps both `id` columns of the join.
    a = spark.createDataFrame([(1, "p")], ["id", "v"])
    b = spark.createDataFrame([(1, "q")], ["id", "w"])
    joined = a.join(b, a.id == b.id)
    assert joined.columns == ["id", "v", "id", "w"]
    dropped = joined.drop("v")
    assert dropped.columns == ["id", "id", "w"]
    assert [tuple(r) for r in dropped.collect()] == [(1, 1, "q")]
    renamed = joined.withColumnsRenamed({"v": "vv"})
    assert renamed.columns == ["id", "vv", "id", "w"]
    assert [tuple(r) for r in renamed.collect()] == [(1, "p", 1, "q")]
    assert joined.withColumnRenamed("w", "ww").columns == ["id", "v", "id", "ww"]
