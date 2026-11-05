# SPDX-License-Identifier: Apache-2.0
"""PySpark's ``Window``/``WindowSpec`` and ``Column.over`` (P2-T190).

Each test follows a PySpark 4.2 documentation example (``Window.partitionBy``,
``Window.orderBy``, ``Window.rowsBetween``, ``Column.over``, ``lag``) with the
expected output from the PySpark docs. Where the engine cannot run something
yet (``RANGE`` frames with value bounds), the test asserts the engine's own
error instead, as the SQL form does.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, SparkSession, Window, WindowSpec
from krishiv.sql import functions as F

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def docs(spark: SparkSession) -> DataFrame:
    """The rows of the ``Window`` docstring examples."""
    return spark.createDataFrame(
        pa.table(
            {
                "id": [1, 1, 2, 1, 2, 3],
                "category": ["a", "a", "a", "b", "b", "b"],
            }
        )
    )


def dicts(df: DataFrame, *sort: str) -> list[dict[str, Any]]:
    rows = [r.asDict() for r in df.collect()]
    if sort:
        rows.sort(key=lambda r: [(r[c] is None, str(r[c])) for c in sort])
    return rows


def vals(df: DataFrame, *sort: str) -> list[Any]:
    """The ``v`` column of ``df``'s rows, in ``sort`` order."""
    return [r["v"] for r in dicts(df, *sort)]


# Constants --------------------------------------------------------------------------------------


def test_window_constants_match_pyspark() -> None:
    assert Window.unboundedPreceding == -(2**63) == -sys.maxsize - 1
    assert Window.unboundedFollowing == 2**63 - 1 == sys.maxsize
    assert Window.currentRow == 0
    assert max(-sys.maxsize, -(2**63)) == Window._PRECEDING_THRESHOLD
    assert min(sys.maxsize, 2**63 - 1) == Window._FOLLOWING_THRESHOLD


# PySpark documentation examples -----------------------------------------------------------------


def test_partition_by_doc_example(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id")
    assert isinstance(window, WindowSpec)
    got = dicts(docs.withColumn("row_number", F.row_number().over(window)), "category", "id", "row_number")
    assert got == [
        {"id": 1, "category": "a", "row_number": 1},
        {"id": 1, "category": "a", "row_number": 2},
        {"id": 2, "category": "a", "row_number": 3},
        {"id": 1, "category": "b", "row_number": 1},
        {"id": 2, "category": "b", "row_number": 2},
        {"id": 3, "category": "b", "row_number": 3},
    ]


def test_order_by_doc_example(docs: DataFrame) -> None:
    window = Window.partitionBy("id").orderBy("category")
    got = dicts(docs.withColumn("row_number", F.row_number().over(window)), "id", "category", "row_number")
    assert got == [
        {"id": 1, "category": "a", "row_number": 1},
        {"id": 1, "category": "a", "row_number": 2},
        {"id": 1, "category": "b", "row_number": 3},
        {"id": 2, "category": "a", "row_number": 1},
        {"id": 2, "category": "b", "row_number": 2},
        {"id": 3, "category": "b", "row_number": 1},
    ]


def test_rows_between_doc_example(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id").rowsBetween(Window.currentRow, 1)
    got = dicts(docs.withColumn("sum", F.sum("id").over(window)), "id", "category", "sum")
    assert got == [
        {"id": 1, "category": "a", "sum": 2},
        {"id": 1, "category": "a", "sum": 3},
        {"id": 1, "category": "b", "sum": 3},
        {"id": 2, "category": "a", "sum": 2},
        {"id": 2, "category": "b", "sum": 5},
        {"id": 3, "category": "b", "sum": 3},
    ]


def test_range_between_doc_example(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id").rangeBetween(Window.currentRow, 1)
    got = dicts(docs.withColumn("sum", F.sum("id").over(window)), "id", "category", "sum")
    assert got == [
        {"id": 1, "category": "a", "sum": 4},
        {"id": 1, "category": "a", "sum": 4},
        {"id": 1, "category": "b", "sum": 3},
        {"id": 2, "category": "a", "sum": 2},
        {"id": 2, "category": "b", "sum": 5},
        {"id": 3, "category": "b", "sum": 3},
    ]


def test_lag_doc_examples(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pa.table({"c1": ["a", "a", "a", "b", "b"], "c2": [1, 2, 3, 8, 2]}),
    )
    window = Window.partitionBy("c1").orderBy("c2")

    def previous(*args: Any) -> list[dict[str, Any]]:
        return dicts(df.withColumn("previous_value", F.lag("c2", *args).over(window)), "c1", "c2")

    assert previous() == [
        {"c1": "a", "c2": 1, "previous_value": None},
        {"c1": "a", "c2": 2, "previous_value": 1},
        {"c1": "a", "c2": 3, "previous_value": 2},
        {"c1": "b", "c2": 2, "previous_value": None},
        {"c1": "b", "c2": 8, "previous_value": 2},
    ]
    assert previous(1, 0) == [
        {"c1": "a", "c2": 1, "previous_value": 0},
        {"c1": "a", "c2": 2, "previous_value": 1},
        {"c1": "a", "c2": 3, "previous_value": 2},
        {"c1": "b", "c2": 2, "previous_value": 0},
        {"c1": "b", "c2": 8, "previous_value": 2},
    ]
    assert previous(2, -1) == [
        {"c1": "a", "c2": 1, "previous_value": -1},
        {"c1": "a", "c2": 2, "previous_value": -1},
        {"c1": "a", "c2": 3, "previous_value": 1},
        {"c1": "b", "c2": 2, "previous_value": -1},
        {"c1": "b", "c2": 8, "previous_value": -1},
    ]


# Ranking functions ------------------------------------------------------------------------------


def test_rank_family(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id")

    def values(name: str, *args: Any) -> list[Any]:
        return vals(docs.withColumn("v", getattr(F, name)(*args).over(window)), "category", "id")

    assert values("rank") == [1, 1, 3, 1, 2, 3]
    assert values("dense_rank") == [1, 1, 2, 1, 2, 3]
    assert values("percent_rank") == [0.0, 0.0, 1.0, 0.0, 0.5, 1.0]
    assert values("cume_dist") == [2 / 3, 2 / 3, 1.0, 1 / 3, 2 / 3, 1.0]
    assert values("ntile", 2) == [1, 1, 2, 1, 1, 2]


# Offset functions --------------------------------------------------------------------------------


def test_lag_and_lead(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id")

    def values(build: Any) -> list[Any]:
        return vals(docs.withColumn("v", build.over(window)), "category", "id")

    assert values(F.lag("id")) == [None, 1, 1, None, 1, 2]
    assert values(F.lead("id")) == [1, 2, None, 2, 3, None]
    assert values(F.lag("id", 2, -1)) == [-1, -1, 1, -1, -1, 1]
    assert values(F.lead("id", 2, -1)) == [2, -1, -1, 3, -1, -1]


def test_lead_takes_a_column_default(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id")
    got = vals(docs.withColumn("v", F.lead("id", 1, F.col("id")).over(window)), "category", "id")
    assert got == [1, 2, 2, 2, 3, 3]


def test_nth_value_first_and_last(spark: SparkSession) -> None:
    df = spark.createDataFrame(pa.table({"id": [1, None, 2, None], "g": ["a", "a", "a", "a"]}))
    window = Window.partitionBy("g").orderBy("id")

    def values(build: Any) -> list[Any]:
        return vals(df.withColumn("v", build.over(window)), "id")

    assert values(F.nth_value("id", 2)) == [None, None, None, None]
    assert values(F.first("id")) == [None, None, None, None]
    assert values(F.first("id", ignorenulls=True)) == [1, 1, None, None]
    assert values(F.last("id")) == [1, 2, None, None]
    assert values(F.last("id", ignorenulls=True)) == [1, 2, None, None]
    assert values(F.first_value("id")) == [None, None, None, None]
    assert values(F.last_value("id")) == [1, 2, None, None]


# Aggregates and default frames -------------------------------------------------------------------


def test_aggregates_over_default_frames(docs: DataFrame) -> None:
    # No ordering: the whole partition (`ROWS UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`).
    whole = Window.partitionBy("category")
    assert vals(docs.withColumn("v", F.sum("id").over(whole)), "category", "id") == [4, 4, 4, 6, 6, 6]
    # Ordered: the running frame (`RANGE UNBOUNDED PRECEDING AND CURRENT ROW`).
    running = Window.partitionBy("category").orderBy("id")
    assert vals(docs.withColumn("v", F.count("id").over(running)), "category", "id") == [2, 2, 3, 1, 2, 3]
    assert vals(docs.withColumn("v", F.min("id").over(running)), "category", "id") == [1, 1, 1, 1, 1, 1]
    assert vals(docs.withColumn("v", F.max("id").over(running)), "category", "id") == [1, 1, 2, 1, 2, 3]


def test_explicit_frames(docs: DataFrame) -> None:
    ordered = Window.partitionBy("category").orderBy("id")
    rows = vals(docs.withColumn("v", F.sum("id").over(ordered.rowsBetween(0, 0))), "category", "id")
    assert rows == [1, 1, 2, 1, 2, 3]
    unbounded = (
        Window.partitionBy("category").orderBy("id").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )
    assert vals(docs.withColumn("v", F.sum("id").over(unbounded)), "category", "id") == [4, 4, 4, 6, 6, 6]
    # No partitioning: one window over all rows.
    everywhere = Window.orderBy("id").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    assert vals(docs.withColumn("v", F.sum("id").over(everywhere)), "id", "category") == [10] * 6
    # An unbounded `RANGE` frame needs no ordering (nothing is computed from the key).
    whole_range = Window.rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    assert vals(docs.withColumn("v", F.sum("id").over(whole_range)), "id", "category") == [10] * 6


def test_out_of_threshold_bounds_are_unbounded(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id").rowsBetween(-(2**70), 2**70).rangeBetween(-(2**70), 2**70)
    assert vals(docs.withColumn("v", F.sum("id").over(window)), "category", "id") == [4, 4, 4, 6, 6, 6]


# Spec semantics ----------------------------------------------------------------------------------


def test_builders_replace_and_return_new_specs(docs: DataFrame) -> None:
    base = Window.partitionBy("category")
    ordered = base.orderBy("id")
    # `base` is unchanged: it still has no ordering, so a ranking fails over it
    # (Spark 4.2 raises `WINDOW_FUNCTION_FRAME_NOT_ORDERED` here; see the
    # `row_number() OVER(PARTITION BY cate)` case in the window.sql golden).
    with pytest.raises(errors.AnalysisException) as excinfo:
        docs.withColumn("v", F.row_number().over(base)).collect()
    assert excinfo.value.getCondition() == "WINDOW_FUNCTION_FRAME_NOT_ORDERED"
    assert excinfo.value.getMessageParameters() == {
        "wf_name": "row_number",
        "wf_expr": "row_number()",
    }
    assert vals(docs.withColumn("v", F.row_number().over(ordered)), "category", "id") == [1, 2, 3, 1, 2, 3]
    # A repeated call replaces: partitioning by `id` sums each id, not each category.
    repart = Window.partitionBy("category").partitionBy("id")
    got = dicts(docs.withColumn("v", F.sum("id").over(repart)), "id", "category")
    assert [r["v"] for r in got] == [3, 3, 3, 4, 4, 3]


def test_partition_and_order_take_a_single_list(docs: DataFrame) -> None:
    window = Window.partitionBy(["category"]).orderBy(["id"])
    assert vals(docs.withColumn("v", F.row_number().over(window)), "category", "id") == [1, 2, 3, 1, 2, 3]


def test_order_by_column_expressions(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy(F.col("id").desc())
    assert [r["v"] for r in dicts(docs.withColumn("v", F.row_number().over(window)), "category", "id")] == [
        2,
        3,
        1,
        3,
        2,
        1,
    ]
    ascending = Window.partitionBy("category").orderBy(F.col("id").asc())
    assert [r["v"] for r in dicts(docs.withColumn("v", F.row_number().over(ascending)), "category", "id")] == [
        1,
        2,
        3,
        1,
        2,
        3,
    ]


# Argument checks ---------------------------------------------------------------------------------


def test_over_requires_a_window_spec() -> None:
    with pytest.raises(errors.PySparkTypeError) as excinfo:
        F.sum("id").over("window")  # ty: ignore[invalid-argument-type]
    assert excinfo.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert excinfo.value.getMessageParameters() == {
        "arg_name": "window",
        "expected_type": "WindowSpec",
        "arg_type": "str",
    }


def test_frame_bounds_require_ints() -> None:
    with pytest.raises(errors.PySparkTypeError) as excinfo:
        Window.rowsBetween("1", 1)  # ty: ignore[invalid-argument-type]
    assert excinfo.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert excinfo.value.getMessageParameters() == {
        "arg_name": "start",
        "expected_type": "int",
        "arg_type": "str",
    }
    with pytest.raises(errors.PySparkTypeError) as excinfo:
        Window.orderBy().rangeBetween(0, 1.5)  # ty: ignore[invalid-argument-type]
    assert excinfo.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert excinfo.value.getMessageParameters() == {
        "arg_name": "end",
        "expected_type": "int",
        "arg_type": "float",
    }


def test_partition_by_rejects_other_types() -> None:
    with pytest.raises(errors.PySparkTypeError) as excinfo:
        Window.partitionBy(1)  # ty: ignore[invalid-argument-type]
    assert excinfo.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert excinfo.value.getMessageParameters() == {
        "arg_name": "cols",
        "expected_type": "Column, str or list of Column or str",
        "arg_type": "int",
    }


def test_signatures_match_pyspark() -> None:
    assert list(inspect.signature(Window.partitionBy).parameters) == ["cols"]
    assert list(inspect.signature(Window.orderBy).parameters) == ["cols"]
    assert list(inspect.signature(Window.rowsBetween).parameters) == ["start", "end"]
    assert list(inspect.signature(Window.rangeBetween).parameters) == ["start", "end"]
    assert list(inspect.signature(WindowSpec.partitionBy).parameters) == ["self", "cols"]
    assert list(inspect.signature(WindowSpec.orderBy).parameters) == ["self", "cols"]
    assert list(inspect.signature(WindowSpec.rowsBetween).parameters) == ["self", "start", "end"]
    assert list(inspect.signature(WindowSpec.rangeBetween).parameters) == ["self", "start", "end"]
    from krishiv.sql.column import Column

    assert list(inspect.signature(Column.over).parameters) == ["self", "window"]
    assert list(inspect.signature(F.lag).parameters) == ["col", "offset", "default"]
    assert list(inspect.signature(F.lead).parameters) == ["col", "offset", "default"]
    assert list(inspect.signature(F.nth_value).parameters) == ["col", "offset", "ignoreNulls"]
    for name in ("partitionBy", "orderBy"):
        for owner in (Window, WindowSpec):
            kind = inspect.signature(getattr(owner, name)).parameters["cols"].kind
            assert kind is inspect.Parameter.VAR_POSITIONAL
    for name in ("rowsBetween", "rangeBetween"):
        for owner in (Window, WindowSpec):
            for param in ("start", "end"):
                found = inspect.signature(getattr(owner, name)).parameters[param]
                assert found.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                assert found.default is inspect.Parameter.empty


# Invalid frames raise the engine's error ----------------------------------------------------------


def test_frame_start_after_end(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id").rowsBetween(1, -1)
    with pytest.raises(errors.AnalysisException) as excinfo:
        docs.withColumn("v", F.sum("id").over(window)).collect()
    assert excinfo.value.getCondition() == "DATATYPE_MISMATCH.SPECIFIED_WINDOW_FRAME_WRONG_COMPARISON"


def test_unbounded_following_lower_bound(docs: DataFrame) -> None:
    window = Window.rowsBetween(Window.unboundedFollowing, Window.unboundedFollowing)
    with pytest.raises(errors.AnalysisException) as excinfo:
        docs.withColumn("v", F.sum("id").over(window)).collect()
    assert excinfo.value.getCondition() == "DATATYPE_MISMATCH.SPECIFIED_WINDOW_FRAME_INVALID_BOUND"


def test_bounds_past_the_int_range(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id").rowsBetween(-(2**40), 2**40)
    with pytest.raises(errors.AnalysisException) as excinfo:
        docs.withColumn("v", F.sum("id").over(window)).collect()
    assert excinfo.value.getCondition() == "DATATYPE_MISMATCH.SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE"


def test_window_function_without_over(docs: DataFrame) -> None:
    with pytest.raises(errors.PySparkException) as excinfo:
        docs.select(F.row_number()).collect()
    assert excinfo.value.getCondition() == "WINDOW_FUNCTION_WITHOUT_OVER_CLAUSE"


def test_window_function_without_ordering(docs: DataFrame) -> None:
    with pytest.raises(errors.PySparkException) as excinfo:
        docs.withColumn("v", F.lag("id").over(Window.partitionBy("category"))).collect()
    assert excinfo.value.getCondition() == "WINDOW_FUNCTION_FRAME_NOT_ORDERED"


def test_ranking_rejects_a_user_frame(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id").rowsBetween(-1, 1)
    with pytest.raises(errors.PySparkException) as excinfo:
        docs.withColumn("v", F.row_number().over(window)).collect()
    assert excinfo.value.getCondition() == "_LEGACY_ERROR_TEMP_1036"


def test_lag_rejects_a_user_frame(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id").rowsBetween(-1, 1)
    with pytest.raises(errors.PySparkException) as excinfo:
        docs.withColumn("v", F.lag("id").over(window)).collect()
    assert excinfo.value.getCondition() == "_LEGACY_ERROR_TEMP_1035"


def test_ntile_needs_a_positive_bucket_count(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id")
    with pytest.raises(errors.AnalysisException) as excinfo:
        docs.withColumn("v", F.ntile(0).over(window)).collect()
    assert excinfo.value.getCondition() == "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE"


def test_nth_value_needs_a_positive_offset(docs: DataFrame) -> None:
    window = Window.partitionBy("category").orderBy("id")
    with pytest.raises(errors.AnalysisException) as excinfo:
        docs.withColumn("v", F.nth_value("id", 0).over(window)).collect()
    assert excinfo.value.getCondition() == "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE"
