# SPDX-License-Identifier: Apache-2.0
"""The relation UDF cases, built against ``pyspark`` or ``krishiv``.

One module builds every case for both engines: ``gen_goldens.py`` runs it on PySpark 4.2.0 (local, a real JVM) and
writes ``goldens.json``; ``test_relation_goldens.py`` runs it on a Krishiv session and compares. The cases cover
``mapInPandas`` / ``mapInArrow``, ``groupBy().applyInPandas`` / ``applyInArrow``, their cogrouped forms and
grouped-aggregate ``pandas_udf`` / ``arrow_udf``: the result checks, ``f(pdf)`` against ``f(key, pdf)``, binding of
result columns by name and by position, a cogroup key on one side only, empty inputs, and several aggregates in one
``agg``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

#: (k, s, v): keys 1 and 2, with a NULL value.
ROWS: list[tuple[Any, ...]] = [(1, "a", 1.0), (1, "b", 2.0), (2, "c", 3.0), (2, "d", None), (2, "e", 10.0)]
SCHEMA = "k long, s string, v double"
#: The cogroup's other side: key 2 is shared, key 3 is on this side only, key 1 on the left only.
OTHER_ROWS: list[tuple[Any, ...]] = [(2, 20), (3, 30), (3, 31)]
OTHER_SCHEMA = "k long, w long"

#: Session conf for every case.
CONF = {"spark.sql.session.timeZone": "UTC"}


class Api:
    """``pyspark`` or ``krishiv`` under one set of names (both packages mirror ``pyspark.sql``)."""

    def __init__(self, root: str) -> None:
        self.F = importlib.import_module(f"{root}.sql.functions")


def _df(spark: Any) -> Any:
    return spark.createDataFrame(ROWS, SCHEMA)


def _other(spark: Any) -> Any:
    return spark.createDataFrame(OTHER_ROWS, OTHER_SCHEMA)


def _empty(spark: Any) -> Any:
    return _df(spark).where("k > 100")


# --- map -------------------------------------------------------------------------------------------------------


def _map_pandas(shape: Callable[[pd.DataFrame], Any], schema: str) -> Callable[..., Any]:
    def build(api: Api, spark: Any) -> Any:
        def f(batches: Any) -> Any:
            for pdf in batches:
                yield shape(pdf)

        return _df(spark).coalesce(1).mapInPandas(f, schema)

    return build


def _map_arrow(shape: Callable[[pa.RecordBatch], Any], schema: str) -> Callable[..., Any]:
    def build(api: Api, spark: Any) -> Any:
        def f(batches: Any) -> Any:
            for batch in batches:
                yield shape(batch)

        return _df(spark).coalesce(1).mapInArrow(f, schema)

    return build


def _map_not_iterator(api: Api, spark: Any) -> Any:
    return _df(spark).coalesce(1).mapInPandas(lambda batches: 1, "k long")


MAP_CASES: dict[str, Callable[..., Any]] = {
    "map/pandas/same_columns": _map_pandas(lambda p: p[["k", "s"]], "k long, s string"),
    "map/pandas/reordered_by_name": _map_pandas(lambda p: p[["s", "k"]], "k long, s string"),
    "map/pandas/extra_column": _map_pandas(lambda p: p[["k", "s"]], "k long"),
    "map/pandas/missing_column": _map_pandas(lambda p: p[["k"]], "k long, other long"),
    "map/pandas/other_names": _map_pandas(lambda p: p[["k", "s"]], "a long, b string"),
    "map/pandas/positional_labels": _map_pandas(lambda p: p[["k", "s"]].set_axis([0, 1], axis=1), "a long, b string"),
    "map/pandas/positional_too_few": _map_pandas(lambda p: p[["k"]].set_axis([0], axis=1), "a long, b string"),
    "map/pandas/wrong_type": _map_pandas(lambda p: p[["s"]], "s long"),
    "map/pandas/returns_series": _map_pandas(lambda p: p["k"], "k long"),
    "map/pandas/not_iterator": _map_not_iterator,
    "map/pandas/empty_result": _map_pandas(lambda p: p.iloc[0:0][["k"]], "k long"),
    "map/pandas/no_columns_and_no_rows": _map_pandas(lambda p: pd.DataFrame(), "k long"),
    "map/arrow/same_columns": _map_arrow(lambda b: b.select(["k", "s"]), "k long, s string"),
    "map/arrow/reordered_by_name": _map_arrow(lambda b: b.select(["s", "k"]), "k long, s string"),
    "map/arrow/extra_column": _map_arrow(lambda b: b.select(["k", "s"]), "k long"),
    "map/arrow/missing_column": _map_arrow(lambda b: b.select(["k"]), "k long, other long"),
    "map/arrow/wrong_type": _map_arrow(lambda b: b.select(["s"]), "s long"),
    "map/arrow/returns_table": _map_arrow(lambda b: pa.Table.from_batches([b.select(["k"])]), "k long"),
}


# --- grouped map -----------------------------------------------------------------------------------------------


def _grouped(
    f: Callable[..., Any], schema: str, *, arrow: bool = False, empty: bool = False, order: tuple[str, ...] = ("k", "n")
) -> Callable[..., Any]:
    def build(api: Api, spark: Any) -> Any:
        grouped = (_empty(spark) if empty else _df(spark)).groupBy("k")
        apply = grouped.applyInArrow if arrow else grouped.applyInPandas
        return apply(f, schema).orderBy(*order)

    return build


def _count(pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"k": [pdf["k"].iloc[0]], "n": [len(pdf)]})


def _key_count(key: Any, pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"k": [key[0]], "n": [len(pdf)]})


def _varargs_count(*args: Any) -> pd.DataFrame:
    return pd.DataFrame({"k": [args[-1]["k"].iloc[0]], "n": [len(args)]})


def _columns_seen(pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"k": [pdf["k"].iloc[0]], "n": [",".join(pdf.columns)]})


def _no_arguments() -> pd.DataFrame:
    return pd.DataFrame({"k": [1], "n": [1]})


def _three_arguments(a: Any, b: Any, c: Any) -> pd.DataFrame:
    return pd.DataFrame({"k": [1], "n": [1]})


def _arrow_count(table: pa.Table) -> pa.Table:
    return pa.table({"k": [table["k"][0].as_py()], "n": [table.num_rows]})


def _arrow_key_count(key: Any, table: pa.Table) -> pa.Table:
    return pa.table({"k": [key[0].as_py()], "n": [table.num_rows]})


def _arrow_reordered(table: pa.Table) -> pa.Table:
    return pa.table({"n": [table.num_rows], "k": [table["k"][0].as_py()]})


GROUPED_CASES: dict[str, Callable[..., Any]] = {
    "grouped/pandas/data_only": _grouped(_count, "k long, n long"),
    "grouped/pandas/key_and_data": _grouped(_key_count, "k long, n long"),
    "grouped/pandas/varargs": _grouped(_varargs_count, "k long, n long"),
    "grouped/pandas/columns_seen": _grouped(_columns_seen, "k long, n string"),
    "grouped/pandas/no_arguments": _grouped(_no_arguments, "k long, n long"),
    "grouped/pandas/three_arguments": _grouped(_three_arguments, "k long, n long"),
    "grouped/pandas/reordered_by_name": _grouped(lambda p: _count(p)[["n", "k"]], "k long, n long"),
    "grouped/pandas/positional_labels": _grouped(lambda p: _count(p).set_axis([0, 1], axis=1), "k long, n long"),
    "grouped/pandas/positional_too_many": _grouped(
        lambda p: _count(p).set_axis([0, 1], axis=1), "n long", order=("n",)
    ),
    "grouped/pandas/extra_column": _grouped(lambda p: _count(p).assign(x=1), "k long, n long"),
    "grouped/pandas/missing_column": _grouped(lambda p: _count(p)[["k"]], "k long, n long"),
    "grouped/pandas/other_names": _grouped(lambda p: _count(p).rename(columns={"n": "m"}), "k long, n long"),
    "grouped/pandas/wrong_type": _grouped(lambda p: _count(p).assign(n="x"), "k long, n long"),
    "grouped/pandas/returns_series": _grouped(lambda p: p["k"], "k long, n long"),
    "grouped/pandas/returns_none": _grouped(lambda p: None, "k long, n long"),
    "grouped/pandas/empty_frame_no_columns": _grouped(lambda p: pd.DataFrame(), "k long, n long"),
    "grouped/pandas/empty_input": _grouped(_count, "k long, n long", empty=True),
    "grouped/arrow/data_only": _grouped(_arrow_count, "k long, n long", arrow=True),
    "grouped/arrow/key_and_data": _grouped(_arrow_key_count, "k long, n long", arrow=True),
    "grouped/arrow/reordered_by_name": _grouped(_arrow_reordered, "k long, n long", arrow=True),
    "grouped/arrow/extra_column": _grouped(
        lambda t: _arrow_count(t).append_column("x", pa.array([1])), "k long, n long", arrow=True
    ),
    "grouped/arrow/missing_column": _grouped(lambda t: _arrow_count(t).select(["k"]), "k long, n long", arrow=True),
    "grouped/arrow/wrong_type": _grouped(lambda t: pa.table({"k": [1], "n": ["x"]}), "k long, n long", arrow=True),
    "grouped/arrow/returns_batch": _grouped(lambda t: _arrow_count(t).to_batches()[0], "k long, n long", arrow=True),
    "grouped/arrow/empty_input": _grouped(_arrow_count, "k long, n long", arrow=True, empty=True),
}


# --- cogrouped map ---------------------------------------------------------------------------------------------


def _cogrouped(f: Callable[..., Any], schema: str, *, arrow: bool = False) -> Callable[..., Any]:
    def build(api: Api, spark: Any) -> Any:
        both = _df(spark).groupBy("k").cogroup(_other(spark).groupBy("k"))
        apply = both.applyInArrow if arrow else both.applyInPandas
        return apply(f, schema).orderBy("k")

    return build


def _sides(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    k = left["k"].iloc[0] if len(left) else right["k"].iloc[0]
    return pd.DataFrame(
        {"k": [k], "sides": [f"{len(left)}:{len(right)}:{','.join(left.columns)}:{','.join(right.columns)}"]}
    )


def _key_sides(key: Any, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"k": [key[0]], "sides": [f"{len(left)}:{len(right)}"]})


def _arrow_sides(left: pa.Table, right: pa.Table) -> pa.Table:
    k = pc.min(left["k"] if left.num_rows else right["k"]).as_py()
    return pa.table({"k": [k], "sides": [f"{left.num_rows}:{right.num_rows}"]})


COGROUPED_CASES: dict[str, Callable[..., Any]] = {
    "cogrouped/pandas/one_side_keys": _cogrouped(_sides, "k long, sides string"),
    "cogrouped/pandas/key_and_data": _cogrouped(_key_sides, "k long, sides string"),
    "cogrouped/pandas/one_argument": _cogrouped(lambda left: left, "k long, sides string"),
    "cogrouped/pandas/missing_column": _cogrouped(lambda a, b: _sides(a, b)[["k"]], "k long, sides string"),
    "cogrouped/arrow/one_side_keys": _cogrouped(_arrow_sides, "k long, sides string", arrow=True),
    "cogrouped/arrow/missing_column": _cogrouped(
        lambda a, b: _arrow_sides(a, b).select(["k"]), "k long, sides string", arrow=True
    ),
}


# --- grouped aggregate -----------------------------------------------------------------------------------------


def _aggs(api: Api) -> dict[str, Any]:
    F = api.F

    @F.pandas_udf("double")
    def total(v: pd.Series) -> float:
        return float(v.sum())

    @F.pandas_udf("long")
    def count(v: pd.Series) -> int:
        return len(v)

    @F.pandas_udf("long")
    def not_a_scalar(v: pd.Series) -> int:
        return v  # type: ignore[return-value]

    @F.arrow_udf("double")
    def arrow_total(v: pa.Array) -> float:
        return pc.sum(v).as_py()

    return {"total": total, "count": count, "not_a_scalar": not_a_scalar, "arrow_total": arrow_total}


def _agg(columns: Callable[[Api, dict[str, Any]], list[Any]], *, keys: bool = True, empty: bool = False) -> Any:
    def build(api: Api, spark: Any) -> Any:
        source = _empty(spark) if empty else _df(spark)
        out = columns(api, _aggs(api))
        if keys:
            return source.groupBy("k").agg(*out).orderBy("k")
        return source.agg(*out)

    return build


AGG_CASES: dict[str, Callable[..., Any]] = {
    "agg/pandas/one": _agg(lambda api, u: [u["total"]("v").alias("t")]),
    "agg/pandas/two": _agg(lambda api, u: [u["total"]("v").alias("t"), u["count"]("v").alias("c")]),
    "agg/pandas/same_twice": _agg(lambda api, u: [u["total"]("v").alias("a"), u["total"]("v").alias("b")]),
    "agg/pandas/global": _agg(lambda api, u: [u["total"]("v").alias("t")], keys=False),
    "agg/pandas/global_empty_input": _agg(lambda api, u: [u["count"]("v").alias("c")], keys=False, empty=True),
    "agg/pandas/grouped_empty_input": _agg(lambda api, u: [u["count"]("v").alias("c")], empty=True),
    "agg/pandas/mixed_with_builtin": _agg(lambda api, u: [u["total"]("v").alias("t"), api.F.max("v").alias("m")]),
    "agg/pandas/not_a_scalar": _agg(lambda api, u: [u["not_a_scalar"]("v").alias("t")]),
    "agg/arrow/one": _agg(lambda api, u: [u["arrow_total"]("v").alias("t")]),
}
# One ``agg`` mixing a pandas and an arrow aggregate has no case: Spark 4.2.0 fails it with an internal assertion
# ("All aggregate functions must have the same eval type in ArrowAggregatePythonExec"), not a behaviour to copy.


# --- declared NOT NULL -----------------------------------------------------------------------------------------


def _null_n(pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"k": [pdf["k"].iloc[0]], "n": pd.array([None], dtype="Int64")})


def _arrow_null_n(table: pa.Table) -> pa.Table:
    return pa.table({"k": [table["k"][0].as_py()], "n": pa.array([None], pa.int64())})


def _sides_null(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return _sides(left, right).assign(sides=None)


NOT_NULL_CASES: dict[str, Callable[..., Any]] = {
    "not_null/map/pandas": _map_pandas(lambda p: p[["v"]], "v double not null"),
    "not_null/map/arrow": _map_arrow(lambda b: b.select(["v"]), "v double not null"),
    "not_null/map/arrow_no_null": _map_arrow(lambda b: b.select(["k"]), "k long not null"),
    "not_null/grouped/pandas": _grouped(_null_n, "k long, n long not null"),
    "not_null/grouped/arrow": _grouped(_arrow_null_n, "k long, n long not null", arrow=True),
    "not_null/cogrouped/pandas": _cogrouped(_sides_null, "k long, sides string not null"),
}


CASES: dict[str, Callable[..., Any]] = {**MAP_CASES, **GROUPED_CASES, **COGROUPED_CASES, **AGG_CASES, **NOT_NULL_CASES}


def cell(value: Any) -> Any:
    """A result cell as JSON: NaN and infinities as strings, everything else as is."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value
