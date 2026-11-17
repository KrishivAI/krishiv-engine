# SPDX-License-Identifier: Apache-2.0
"""The relation kinds of Python UDFs (``krishiv._udf.relations``; design ``P3-T085-embedded-python-udf-kinds.md``
§3.2), run in this process over a scripted frame source.

The server runs the same runners in a pool worker (``krishiv-udf-pool``'s ``tests/pool.rs`` drives real workers
with ``pyspark-client``); that interpreter is not measured, so this file runs each kind here: the frames a runner
reads, the batches it yields, and the error class of every way a UDF or the frames can go wrong (Spark's
``UDF_RETURN_TYPE``, ``RESULT_COLUMN_NAMES_MISMATCH``, ... as ``worker.py`` raises them).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

import krishiv.errors as kerrors  # noqa: E402
from krishiv._udf import _host, relations  # noqa: E402
from krishiv._udf import types as udf_types  # noqa: E402
from krishiv._udf.relations import (  # noqa: E402
    SQL_COGROUPED_MAP_ARROW_UDF,
    SQL_COGROUPED_MAP_PANDAS_UDF,
    SQL_GROUPED_AGG_ARROW_ITER_UDF,
    SQL_GROUPED_AGG_ARROW_UDF,
    SQL_GROUPED_AGG_PANDAS_ITER_UDF,
    SQL_GROUPED_AGG_PANDAS_UDF,
    SQL_GROUPED_MAP_ARROW_ITER_UDF,
    SQL_GROUPED_MAP_ARROW_UDF,
    SQL_GROUPED_MAP_PANDAS_ITER_UDF,
    SQL_GROUPED_MAP_PANDAS_UDF,
    SQL_MAP_ARROW_ITER_UDF,
    SQL_MAP_PANDAS_ITER_UDF,
    FrameError,
)
from krishiv.sql.types import LongType, StringType, StructField, StructType  # noqa: E402

KV = StructType([StructField("k", LongType()), StructField("v", LongType())])
V = StructType([StructField("v", LongType())])


class Source:
    """A scripted frame source: the frames in order; asking past the last one is a test bug."""

    def __init__(self, frames: list[tuple[Any, ...]]) -> None:
        self.frames = list(frames)

    def next(self) -> tuple[Any, ...]:
        return self.frames.pop(0)


@pytest.fixture(autouse=True)
def conf(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The runner conf of the running call (the native module's, in-process), a dict the test can edit."""
    values = {"spark.sql.session.timeZone": "UTC"}
    monkeypatch.setattr(_host, "runner_conf", lambda: values)
    return values


def rb(**columns: list[Any]) -> Any:
    return pa.RecordBatch.from_pydict(columns)


def batch(side: int, **columns: list[Any]) -> tuple[Any, ...]:
    return ("batch", side, rb(**columns))


END_GROUP = ("end_group",)
END = ("end",)


def run(
    eval_type: int, f: Any, return_type: Any, frames: list[tuple[Any, ...]], key_count: int = 0, arity: int = 0
) -> list[Any]:
    runner = relations.make_relation(f, eval_type, return_type)
    return list(runner(Source(frames), key_count, arity))


def rows(batches: list[Any]) -> list[dict[str, Any]]:
    return [row for b in batches for row in b.to_pylist()]


def params(err: pytest.ExceptionInfo[Any]) -> dict[str, str]:
    """The error's message parameters (in-process, the facade's errors print only their class)."""
    found = err.value.getMessageParameters()
    assert found is not None
    return found


def message(err: pytest.ExceptionInfo[Any]) -> str:
    return str(params(err))


def error_class(err: pytest.ExceptionInfo[Any]) -> str:
    return err.value.getCondition()


# The exception class ``relations.py`` raises each error class as (``worker.py``'s): a wrong kind of result is a
# ``PySparkTypeError``, a result whose columns do not match the declared schema a ``PySparkRuntimeError``.
RAISED: dict[str, type[kerrors.PySparkException]] = {
    "UDF_RETURN_TYPE": kerrors.PySparkTypeError,
    "RESULT_COLUMN_NAMES_MISMATCH": kerrors.PySparkRuntimeError,
    "RESULT_COLUMN_SCHEMA_MISMATCH": kerrors.PySparkRuntimeError,
    "RESULT_COLUMN_TYPES_MISMATCH": kerrors.PySparkRuntimeError,
}


# --- the frames ---
def test_a_map_task_reads_batches_up_to_the_end() -> None:
    frames = [batch(0, v=[1]), batch(0, v=[2, 3]), END]
    assert [b.num_rows for b in relations._map_batches(Source(frames))] == [1, 2]


def test_a_map_task_refuses_a_group_marker() -> None:
    with pytest.raises(FrameError, match="a map task got 'end_group'"):
        list(relations._map_batches(Source([END_GROUP])))


def test_a_group_starts_with_a_batch() -> None:
    with pytest.raises(FrameError, match="a group starts with 'end_group'"):
        list(relations._groups(Source([END_GROUP])))


def test_groups_are_drained_before_the_next_starts() -> None:
    frames = [batch(0, v=[1]), batch(0, v=[2]), END_GROUP, batch(0, v=[3]), END_GROUP, END]
    seen = []
    for group in relations._groups(Source(frames)):
        seen.append(next(group)[1].column(0).to_pylist())  # the UDF left the rest of the group unread
    assert seen == [[1], [3]]


def test_a_group_cut_by_the_end_is_a_frame_error() -> None:
    with pytest.raises(FrameError, match="the input ended inside a group"):
        list(relations._groups(Source([batch(0, v=[1]), END])))


def test_a_cogroup_needs_each_side_in_order() -> None:
    def sides(frames: list[tuple[Any, ...]]) -> Any:
        group = next(relations._groups(Source([*frames, END_GROUP, END])))
        return relations._sides(group, "UTC")

    left, right = sides([batch(0, v=[1]), batch(0, v=[2]), batch(1, v=[3])])
    assert (len(left), len(right)) == (2, 1)
    with pytest.raises(FrameError, match="side 0 out of order"):
        sides([batch(0, v=[1]), batch(1, v=[2]), batch(0, v=[3])])
    with pytest.raises(FrameError, match="side 2 out of order"):
        sides([batch(2, v=[1])])
    with pytest.raises(FrameError, match="without a frame for each side"):
        sides([batch(0, v=[1])])


def test_a_batch_without_columns_is_left_alone() -> None:
    empty = pa.RecordBatch.from_pydict({"_": [None, None]}).select([])
    assert relations._spark_batch(empty, "UTC") is empty


def test_string_views_reach_the_udf_as_strings_and_come_back_as_views() -> None:
    seen: list[Any] = []

    def f(batches: Iterator[Any]) -> Iterator[Any]:
        for b in batches:
            seen.append(b.schema.field("s").type)
            yield b

    view = pa.record_batch([pa.array(["a", "b"], pa.string_view())], names=["s"])
    out = run(SQL_MAP_ARROW_ITER_UDF, f, StructType([StructField("s", StringType())]), [("batch", 0, view), END])
    assert seen == [pa.string()]
    assert out[0].schema.field("s").type == pa.string_view()
    assert out[0].column(0).to_pylist() == ["a", "b"]


def test_a_large_var_types_conf_makes_large_declared_types(conf: dict[str, str]) -> None:
    conf["spark.sql.execution.arrow.useLargeVarTypes"] = "true"
    schema = relations._engine_schema(StringType(), relations._runner_conf())
    assert schema.names == ["_0"]
    assert schema.field("_0").type == pa.string_view()


# --- map kinds ---
def test_map_in_pandas() -> None:
    def f(frames: Iterator[Any]) -> Iterator[Any]:
        for df in frames:
            yield df.assign(v=df["v"] * 10)

    frames = [batch(0, k=[1, 2], v=[1, 2]), batch(0, k=[3], v=[3]), END]
    out = run(SQL_MAP_PANDAS_ITER_UDF, f, KV, frames)
    assert rows(out) == [{"k": 1, "v": 10}, {"k": 2, "v": 20}, {"k": 3, "v": 30}]


def test_map_in_pandas_with_a_scalar_return_type_yields_series() -> None:
    def f(frames: Iterator[Any]) -> Iterator[Any]:
        for df in frames:
            yield df["v"] + 1

    out = run(SQL_MAP_PANDAS_ITER_UDF, f, LongType(), [batch(0, v=[1, 2]), END])
    assert rows(out) == [{"_0": 2}, {"_0": 3}]


@pytest.mark.parametrize(
    ("f", "return_type", "klass", "text"),
    [
        (lambda frames: 5, KV, "UDF_RETURN_TYPE", "iterator of pandas.DataFrame"),
        (lambda frames: iter([5]), KV, "UDF_RETURN_TYPE", "iterator of int"),
        (lambda frames: iter([pd.Series([1])]), KV, "UDF_RETURN_TYPE", "iterator of Series"),
        (lambda frames: iter([pd.DataFrame({"v": [1]})]), LongType(), "UDF_RETURN_TYPE", "iterator of pandas.Series"),
        (
            lambda frames: iter([pd.DataFrame({"k": [1], "z": [1]})]),
            KV,
            "RESULT_COLUMN_NAMES_MISMATCH",
            "Unexpected: z.",
        ),
        (
            lambda frames: iter([pd.DataFrame({0: [1]})]),
            KV,
            "RESULT_COLUMN_SCHEMA_MISMATCH",
            "",
        ),
    ],
)
def test_map_in_pandas_checks_what_the_udf_returns(f: Any, return_type: Any, klass: str, text: str) -> None:
    with pytest.raises(RAISED[klass]) as err:
        run(SQL_MAP_PANDAS_ITER_UDF, f, return_type, [batch(0, k=[1], v=[2]), END])
    assert error_class(err) == klass
    assert text in message(err)


def test_a_result_with_more_numbered_columns_is_cut_to_the_declared_ones() -> None:
    # `truncate_return_schema=True` for map: only the first columns count when the names are not strings.
    def f(frames: Iterator[Any]) -> Iterator[Any]:
        for df in frames:
            yield pd.DataFrame({0: df["k"], 1: df["v"]})

    out = run(SQL_MAP_PANDAS_ITER_UDF, f, KV, [batch(0, k=[1], v=[2]), END])
    assert rows(out) == [{"k": 1, "v": 2}]


def test_a_stop_iteration_in_a_udf_is_a_runtime_error() -> None:
    def f(frames: Iterator[Any]) -> Iterator[Any]:
        raise StopIteration("done")

    with pytest.raises(kerrors.PySparkRuntimeError) as err:
        run(SQL_MAP_PANDAS_ITER_UDF, f, KV, [batch(0, k=[1], v=[2]), END])
    assert error_class(err) == "STOP_ITERATION_OCCURRED"


def test_map_in_arrow_keeps_the_names_the_udf_returned() -> None:
    def f(batches: Iterator[Any]) -> Iterator[Any]:
        for b in batches:
            yield pa.record_batch([b.column("v"), b.column("v")], names=["x", "y"])

    out = run(SQL_MAP_ARROW_ITER_UDF, f, V, [batch(0, v=[1, 2]), END])
    # the server's check of the result sees the UDF's own columns, the extra one included
    assert out[0].schema.names == ["x", "y"]
    assert rows(out) == [{"x": 1, "y": 1}, {"x": 2, "y": 2}]


def test_map_in_arrow_with_no_columns_keeps_the_row_count() -> None:
    def f(batches: Iterator[Any]) -> Iterator[Any]:
        for b in batches:
            yield pa.RecordBatch.from_pydict({"_": [None] * b.num_rows}).select([])

    out = run(SQL_MAP_ARROW_ITER_UDF, f, StructType([]), [batch(0, v=[1, 2, 3]), END])
    assert [(b.num_columns, b.num_rows) for b in out] == [(0, 3)]


def test_map_in_arrow_leaves_a_column_of_another_type_for_the_servers_check() -> None:
    def f(batches: Iterator[Any]) -> Iterator[Any]:
        for b in batches:
            yield pa.record_batch([b.column("v").cast(pa.float64())], names=["v"])

    out = run(SQL_MAP_ARROW_ITER_UDF, f, V, [batch(0, v=[1]), END])
    assert out[0].schema.field("v").type == pa.float64()


@pytest.mark.parametrize(
    ("f", "expected", "actual"),
    [(lambda batches: 5, "iterator of pyarrow.RecordBatch", "int"), (lambda b: iter([5]), "", "iterator of int")],
)
def test_map_in_arrow_checks_what_the_udf_returns(f: Any, expected: str, actual: str) -> None:
    with pytest.raises(kerrors.PySparkTypeError) as err:
        run(SQL_MAP_ARROW_ITER_UDF, f, V, [batch(0, v=[1]), END])
    assert error_class(err) == "UDF_RETURN_TYPE"
    assert params(err)["actual"] == actual
    assert expected in params(err)["expected"]


# --- grouped map ---
def _groups_frames() -> list[tuple[Any, ...]]:
    # two groups (keys 1 and 2), the first in two batches; a key column leads each frame
    return [
        batch(0, k=[1, 1], v=[10, 20]),
        batch(0, k=[1], v=[30]),
        END_GROUP,
        batch(0, k=[2], v=[5]),
        END_GROUP,
        END,
    ]


def test_grouped_map_in_pandas_with_and_without_the_key() -> None:
    def f1(df: Any) -> Any:
        return pd.DataFrame({"k": [len(df.columns)], "v": [df["v"].sum()]})

    def f2(key: Any, df: Any) -> Any:
        return pd.DataFrame({"k": [key[0]], "v": [df["v"].sum()]})

    expected = [{"k": 1, "v": 60}, {"k": 2, "v": 5}]
    out = run(SQL_GROUPED_MAP_PANDAS_UDF, f2, KV, _groups_frames(), key_count=1)
    assert rows(out) == expected
    # the 1-argument form sees the values only, not the key column
    out = run(SQL_GROUPED_MAP_PANDAS_UDF, f1, KV, _groups_frames(), key_count=1)
    assert rows(out) == [{"k": 1, "v": 60}, {"k": 1, "v": 5}]


@pytest.mark.parametrize(
    ("result", "klass"),
    [
        (5, "UDF_RETURN_TYPE"),
        (pd.DataFrame({"k": [1], "z": [1]}), "RESULT_COLUMN_NAMES_MISMATCH"),
        (pd.DataFrame({0: [1]}), "RESULT_COLUMN_SCHEMA_MISMATCH"),
    ],
)
def test_grouped_map_in_pandas_checks_what_the_udf_returns(result: Any, klass: str) -> None:
    with pytest.raises(RAISED[klass]) as err:
        run(SQL_GROUPED_MAP_PANDAS_UDF, lambda df: result, KV, _groups_frames(), key_count=1)
    assert error_class(err) == klass


def test_grouped_map_in_pandas_with_a_scalar_return_type_refuses_a_series_check() -> None:
    with pytest.raises(kerrors.PySparkTypeError) as err:
        run(SQL_GROUPED_MAP_PANDAS_UDF, lambda df: 5, LongType(), _groups_frames(), key_count=1)
    assert params(err) == {"expected": "pandas.Series", "actual": "int"}


def test_grouped_map_in_pandas_iterator_form() -> None:
    def f1(frames: Iterator[Any]) -> Iterator[Any]:
        total = 0
        for df in frames:
            total += int(df["v"].sum())
        yield pd.DataFrame({"k": [0], "v": [total]})

    def f2(key: Any, frames: Iterator[Any]) -> Iterator[Any]:
        for df in frames:
            yield pd.DataFrame({"k": [key[0]] * len(df), "v": df["v"].to_numpy()})

    out = run(SQL_GROUPED_MAP_PANDAS_ITER_UDF, f1, KV, _groups_frames(), key_count=1)
    assert rows(out) == [{"k": 0, "v": 60}, {"k": 0, "v": 5}]
    out = run(SQL_GROUPED_MAP_PANDAS_ITER_UDF, f2, KV, _groups_frames(), key_count=1)
    assert [r["k"] for r in rows(out)] == [1, 1, 1, 2]
    assert [r["v"] for r in rows(out)] == [10, 20, 30, 5]


def test_grouped_map_in_pandas_iterator_form_refuses_other_arities() -> None:
    def f3(a: Any, b: Any, c: Any) -> Any:
        return iter([])

    # Spark's worker.py calls the UDF in an if/elif on the arity with no else, so any other arity fails reading the
    # unbound `result`; relations.py ports that as is (its UnboundLocalError, relations.py:612, 703, 769).
    with pytest.raises(UnboundLocalError, match="cannot access local variable 'result'"):
        run(SQL_GROUPED_MAP_PANDAS_ITER_UDF, f3, KV, _groups_frames(), key_count=1)


def test_grouped_map_in_pandas_iterator_form_checks_each_frame() -> None:
    def f(frames: Iterator[Any]) -> Iterator[Any]:
        yield pd.DataFrame({"k": [1], "z": [1]})

    with pytest.raises(kerrors.PySparkRuntimeError) as err:
        run(SQL_GROUPED_MAP_PANDAS_ITER_UDF, f, KV, _groups_frames(), key_count=1)
    assert error_class(err) == "RESULT_COLUMN_NAMES_MISMATCH"


def _table_udf(table: Any) -> Any:
    return pa.table({"k": pa.array([0] * table.num_rows, pa.int64()), "v": table.column("v")})


def test_grouped_map_in_arrow() -> None:
    def f1(table: Any) -> Any:
        return pa.table({"k": [0], "v": [sum(table.column("v").to_pylist())]})

    def f2(key: Any, table: Any) -> Any:
        return pa.table({"v": [sum(table.column("v").to_pylist())], "k": [key[0]]})  # out of order: bound by name

    out = run(SQL_GROUPED_MAP_ARROW_UDF, f1, KV, _groups_frames(), key_count=1)
    assert rows(out) == [{"k": 0, "v": 60}, {"k": 0, "v": 5}]
    out = run(SQL_GROUPED_MAP_ARROW_UDF, f2, KV, _groups_frames(), key_count=1)
    assert out[0].schema.names == ["k", "v"]
    assert rows(out) == [{"k": 1, "v": 60}, {"k": 2, "v": 5}]


def test_grouped_map_in_arrow_binds_by_position_when_asked(conf: dict[str, str]) -> None:
    conf["spark.sql.legacy.execution.pandas.groupedMap.assignColumnsByName"] = "false"
    out = run(
        SQL_GROUPED_MAP_ARROW_UDF,
        lambda t: pa.table({"a": [1], "b": [2]}),
        KV,
        _groups_frames(),
        key_count=1,
    )
    assert out[0].schema.names == ["k", "v"]
    assert rows(out) == [{"k": 1, "v": 2}, {"k": 1, "v": 2}]


@pytest.mark.parametrize(
    ("result", "klass", "text"),
    [
        (5, "UDF_RETURN_TYPE", "pyarrow.Table"),
        (pa.table({"k": [1], "z": [1]}), "RESULT_COLUMN_NAMES_MISMATCH", "Unexpected: z."),
        (pa.table({"k": [1], "v": ["x"]}), "RESULT_COLUMN_TYPES_MISMATCH", "column 'v' (expected int64"),
    ],
)
def test_grouped_map_in_arrow_checks_what_the_udf_returns(result: Any, klass: str, text: str) -> None:
    with pytest.raises(RAISED[klass]) as err:
        run(SQL_GROUPED_MAP_ARROW_UDF, lambda t: result, KV, _groups_frames(), key_count=1)
    assert error_class(err) == klass
    assert text in message(err)


def test_grouped_map_in_arrow_accepts_an_empty_result() -> None:
    out = run(SQL_GROUPED_MAP_ARROW_UDF, lambda t: pa.table({}), KV, _groups_frames(), key_count=1)
    assert out == []


def test_grouped_map_in_arrow_iterator_form() -> None:
    def f1(batches: Iterator[Any]) -> Iterator[Any]:
        for b in batches:
            yield pa.record_batch([pa.array([0] * b.num_rows, pa.int64()), b.column("v")], names=["k", "v"])

    def f2(key: Any, batches: Iterator[Any]) -> Iterator[Any]:
        first = next(batches)  # the rest of the group is drained after the UDF returns
        yield pa.record_batch([pa.array([key[0]], pa.int64()), first.column("v")[:1]], names=["k", "v"])

    out = run(SQL_GROUPED_MAP_ARROW_ITER_UDF, f1, KV, _groups_frames(), key_count=1)
    assert [r["v"] for r in rows(out)] == [10, 20, 30, 5]
    out = run(SQL_GROUPED_MAP_ARROW_ITER_UDF, f2, KV, _groups_frames(), key_count=1)
    assert rows(out) == [{"k": 1, "v": 10}, {"k": 2, "v": 5}]


@pytest.mark.parametrize(
    ("element", "klass"),
    [(5, "UDF_RETURN_TYPE"), (pa.record_batch([pa.array([1])], names=["z"]), "RESULT_COLUMN_NAMES_MISMATCH")],
)
def test_grouped_map_in_arrow_iterator_form_checks_each_batch(element: Any, klass: str) -> None:
    with pytest.raises(RAISED[klass]) as err:
        run(SQL_GROUPED_MAP_ARROW_ITER_UDF, lambda it: iter([element]), KV, _groups_frames(), key_count=1)
    assert error_class(err) == klass


def test_arrow_group_kinds_need_a_struct_return_type() -> None:
    for eval_type in (SQL_GROUPED_MAP_ARROW_UDF, SQL_GROUPED_MAP_ARROW_ITER_UDF, SQL_COGROUPED_MAP_ARROW_UDF):
        with pytest.raises(kerrors.PySparkNotImplementedError) as err:
            relations.make_relation(lambda t: t, eval_type, LongType())
        assert params(err) == {"feature": f"Python UDFs of eval type {eval_type} as a relation"}


def test_an_unknown_relation_eval_type_is_not_implemented() -> None:
    with pytest.raises(kerrors.PySparkNotImplementedError) as err:
        relations.make_relation(lambda: None, 999, LongType())
    assert params(err) == {"feature": "Python UDFs of eval type 999 as a relation"}


# --- cogrouped map ---
def _cogroup_frames() -> list[tuple[Any, ...]]:
    # key 1 on both sides (the left in two batches); key 2 only on the right (an empty left batch)
    return [
        batch(0, k=[1], v=[1]),
        batch(0, k=[1], v=[2]),
        batch(1, k=[1], v=[100]),
        END_GROUP,
        batch(0, k=pa.array([], pa.int64()).to_pylist(), v=[]),
        batch(1, k=[2], v=[200]),
        END_GROUP,
        END,
    ]


def test_cogrouped_map_in_pandas() -> None:
    def f2(left: Any, right: Any) -> Any:
        return pd.DataFrame({"k": [0], "v": [int(left["v"].sum() + right["v"].sum())]})

    def f3(key: Any, left: Any, right: Any) -> Any:
        return pd.DataFrame({"k": [key[0]], "v": [len(left) * 1000 + len(right)]})

    out = run(SQL_COGROUPED_MAP_PANDAS_UDF, f2, KV, _cogroup_frames(), key_count=1)
    assert rows(out) == [{"k": 0, "v": 103}, {"k": 0, "v": 200}]
    out = run(SQL_COGROUPED_MAP_PANDAS_UDF, f3, KV, _cogroup_frames(), key_count=1)
    # the key is the right side's where the left has no rows
    assert rows(out) == [{"k": 1, "v": 2001}, {"k": 2, "v": 1}]


def test_cogrouped_map_in_pandas_refuses_other_arities_and_bad_results() -> None:
    def f4(a: Any, b: Any, c: Any, d: Any) -> Any:
        return None

    # Spark's worker.py calls the UDF in an if/elif on the arity with no else, so any other arity fails reading the
    # unbound `result`; relations.py ports that as is (its UnboundLocalError, relations.py:612, 703, 769).
    with pytest.raises(UnboundLocalError):
        run(SQL_COGROUPED_MAP_PANDAS_UDF, f4, KV, _cogroup_frames(), key_count=1)
    with pytest.raises(kerrors.PySparkTypeError) as err:
        run(SQL_COGROUPED_MAP_PANDAS_UDF, lambda left, right: 5, KV, _cogroup_frames(), key_count=1)
    assert error_class(err) == "UDF_RETURN_TYPE"


def test_cogrouped_map_in_arrow() -> None:
    def f2(left: Any, right: Any) -> Any:
        return pa.table({"v": [sum(left.column("v").to_pylist()) + sum(right.column("v").to_pylist())], "k": [0]})

    def f3(key: Any, left: Any, right: Any) -> Any:
        return pa.table({"k": [key[0]], "v": [left.num_rows * 1000 + right.num_rows]})

    out = run(SQL_COGROUPED_MAP_ARROW_UDF, f2, KV, _cogroup_frames(), key_count=1)
    assert rows(out) == [{"k": 0, "v": 103}, {"k": 0, "v": 200}]
    out = run(SQL_COGROUPED_MAP_ARROW_UDF, f3, KV, _cogroup_frames(), key_count=1)
    assert rows(out) == [{"k": 1, "v": 2001}, {"k": 2, "v": 1}]


def test_cogrouped_map_in_arrow_refuses_other_arities_and_bad_results() -> None:
    def f1(a: Any) -> Any:
        return None

    # Spark's worker.py calls the UDF in an if/elif on the arity with no else, so any other arity fails reading the
    # unbound `result`; relations.py ports that as is (its UnboundLocalError, relations.py:612, 703, 769).
    with pytest.raises(UnboundLocalError):
        run(SQL_COGROUPED_MAP_ARROW_UDF, f1, KV, _cogroup_frames(), key_count=1)
    with pytest.raises(kerrors.PySparkTypeError) as err:
        run(SQL_COGROUPED_MAP_ARROW_UDF, lambda left, right: 5, KV, _cogroup_frames(), key_count=1)
    assert error_class(err) == "UDF_RETURN_TYPE"


# --- grouped aggregate ---
def _agg_frames() -> list[tuple[Any, ...]]:
    return [
        batch(0, a=[1, 2], b=[10, 20]),
        batch(0, a=[3], b=[30]),
        END_GROUP,
        batch(0, a=[4], b=[40]),
        END_GROUP,
        END,
    ]


def test_grouped_aggregate_pandas() -> None:
    out = run(SQL_GROUPED_AGG_PANDAS_UDF, lambda s: int(s.sum()), LongType(), _agg_frames(), arity=1)
    assert rows(out) == [{"_0": 6}, {"_0": 4}]
    out = run(SQL_GROUPED_AGG_PANDAS_UDF, lambda a, b: int((a * b).sum()), LongType(), _agg_frames(), arity=2)
    assert rows(out) == [{"_0": 1 * 10 + 2 * 20 + 3 * 30}, {"_0": 160}]


def test_grouped_aggregate_pandas_iterator_form() -> None:
    def one(series: Iterator[Any]) -> int:
        return sum(int(s.sum()) for s in series)

    def two(pairs: Iterator[Any]) -> int:
        return sum(int((a * b).sum()) for a, b in pairs)

    out = run(SQL_GROUPED_AGG_PANDAS_ITER_UDF, one, LongType(), _agg_frames(), arity=1)
    assert rows(out) == [{"_0": 6}, {"_0": 4}]
    out = run(SQL_GROUPED_AGG_PANDAS_ITER_UDF, two, LongType(), _agg_frames(), arity=2)
    assert rows(out) == [{"_0": 140}, {"_0": 160}]


def test_grouped_aggregate_arrow() -> None:
    def total(a: Any) -> int:
        return sum(a.to_pylist())

    def dot(a: Any, b: Any) -> int:
        return sum(x * y for x, y in zip(a.to_pylist(), b.to_pylist(), strict=True))

    out = run(SQL_GROUPED_AGG_ARROW_UDF, total, LongType(), _agg_frames(), arity=1)
    assert rows(out) == [{"_0": 6}, {"_0": 4}]
    out = run(SQL_GROUPED_AGG_ARROW_UDF, dot, LongType(), _agg_frames(), arity=2)
    assert rows(out) == [{"_0": 140}, {"_0": 160}]


def test_grouped_aggregate_arrow_without_concat_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    # pyarrow before 19 has no `concat_batches`: the batches are joined through a struct array
    monkeypatch.delattr(pa, "concat_batches")
    out = run(SQL_GROUPED_AGG_ARROW_UDF, lambda a: sum(a.to_pylist()), LongType(), _agg_frames(), arity=1)
    assert rows(out) == [{"_0": 6}, {"_0": 4}]


def test_grouped_aggregate_arrow_iterator_form() -> None:
    def first_only(batches: Iterator[Any]) -> int:
        return sum(next(batches).to_pylist())  # the rest of the group is drained after the UDF returns

    def pairs(batches: Iterator[Any]) -> int:
        return sum(sum(a.to_pylist()) + sum(b.to_pylist()) for a, b in batches)

    out = run(SQL_GROUPED_AGG_ARROW_ITER_UDF, first_only, LongType(), _agg_frames(), arity=1)
    assert rows(out) == [{"_0": 3}, {"_0": 4}]
    out = run(SQL_GROUPED_AGG_ARROW_ITER_UDF, pairs, LongType(), _agg_frames(), arity=2)
    assert rows(out) == [{"_0": 6 + 60}, {"_0": 44}]


# --- the Spark checks, on their own ---
def test_verify_result_checks_iterability_and_elements() -> None:
    check = relations.verify_result(pa.RecordBatch)
    assert list(check(iter([]))) == []
    with pytest.raises(kerrors.PySparkTypeError) as err:
        check(5)
    assert params(err)["actual"] == "int"
    with pytest.raises(kerrors.PySparkTypeError) as err:
        list(check([1]))
    assert params(err)["actual"] == "iterator of int"


def test_verify_arrow_checks_the_container_type() -> None:
    expected = {"v": pa.int64()}
    with pytest.raises(kerrors.PySparkTypeError) as err:
        relations.verify_arrow_table(rb(v=[1]), True, expected)
    assert params(err) == {"expected": "pyarrow.Table", "actual": "RecordBatch"}
    with pytest.raises(kerrors.PySparkTypeError) as err:
        relations.verify_arrow_batch(pa.table({"v": [1]}), True, expected)
    assert params(err) == {"expected": "pyarrow.RecordBatch", "actual": "Table"}
    relations.verify_arrow_batch(rb(v=[1]), True, expected)  # right


def test_verify_arrow_result_by_position_names_the_mismatched_columns() -> None:
    with pytest.raises(kerrors.PySparkRuntimeError) as err:
        relations.verify_arrow_result(pa.table({"x": ["s"]}), False, [("v", pa.int64())])
    assert error_class(err) == "RESULT_COLUMN_TYPES_MISMATCH"
    assert "column 'v' (expected int64, actual string)" in message(err)


def test_the_relation_runner_without_a_struct_stand_in() -> None:
    # `StructType` is the facade's in-process (the pickled return type of a pool worker is pyspark's own)
    assert relations.StructType is udf_types.StructType
