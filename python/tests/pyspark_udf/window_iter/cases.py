# SPDX-License-Identifier: Apache-2.0
"""The window-aggregate and scalar-iterator UDF cases, built against ``pyspark`` or ``krishiv``.

One module builds every case for both engines: ``gen_goldens.py`` runs it on PySpark 4.2.0 (local, a real JVM) and
writes ``goldens.json``; ``test_window_iter_goldens.py`` runs it on a Krishiv session and compares. The cases cover
the Python eval types 203 (``SQL_WINDOW_AGG_PANDAS_UDF``: a grouped-aggregate ``pandas_udf`` over a window), 253
(``SQL_WINDOW_AGG_ARROW_UDF``: the ``arrow_udf`` one), 204 (``SQL_SCALAR_PANDAS_ITER_UDF``) and 251
(``SQL_SCALAR_ARROW_ITER_UDF``):

- window aggregates over seven window specs: the whole partition, the default frame of an ordered window (a RANGE
  frame up to the current row, so ties share a value), both-unbounded ROWS, a sliding ROWS frame, a RANGE frame,
  current row to the end, and a frame past the current row that is empty for the last rows; with a NULL value and a
  partition whose only value is NULL. Spark sends an unbounded frame as the whole partition once and a bounded one
  row by row (``window_bound_types``), so both paths are covered, and one case per kind mixes them in one projection.
- scalar iterators across several Arrow batches (``spark.sql.execution.arrow.maxRecordsPerBatch`` = 2): a
  well-behaved UDF, one that re-batches its output, and the row checks of pyspark's ``worker.py``: more output
  than input, input not fully consumed, fewer results than inputs, a result that is not an iterator.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

#: The window input: (id, g, o, v). ``id`` orders ROWS frames uniquely; ``o`` has ties (a: 2, 2) for RANGE frames;
#: ``v`` has a NULL inside partition ``a`` and partition ``c`` holds only a NULL.
WINDOW_ROWS: list[tuple[Any, ...]] = [
    (1, "a", 1, 1.0),
    (2, "a", 2, 2.0),
    (3, "a", 2, 4.0),
    (4, "a", 3, None),
    (5, "a", 5, 8.0),
    (6, "b", 1, 16.0),
    (7, "b", 4, 32.0),
    (8, "c", 1, None),
]
WINDOW_SCHEMA = "id long, g string, o long, v double"

#: The scalar-iterator input: five rows, one NULL, in batches of two.
ITER_ROWS: list[tuple[Any, ...]] = [(1, 10), (2, 20), (3, None), (4, 40), (5, 50)]
ITER_SCHEMA = "id long, x long"

#: Session conf for every case.
CONF = {
    "spark.sql.execution.arrow.maxRecordsPerBatch": "2",
    "spark.sql.session.timeZone": "UTC",
}


class Api:
    """``pyspark`` or ``krishiv`` under one set of names (both packages mirror ``pyspark.sql``)."""

    def __init__(self, root: str) -> None:
        self.F = importlib.import_module(f"{root}.sql.functions")
        self.Window = importlib.import_module(f"{root}.sql.window").Window
        self.T = importlib.import_module(f"{root}.sql.types")


def _window_specs(api: Api) -> dict[str, Any]:
    w = api.Window
    by_g = w.partitionBy("g")
    ordered = by_g.orderBy("o", "id")
    return {
        "partition": by_g,
        "ordered_default": by_g.orderBy("o"),
        "rows_unbounded": ordered.rowsBetween(w.unboundedPreceding, w.unboundedFollowing),
        "rows_sliding": ordered.rowsBetween(-1, 1),
        "range_back": by_g.orderBy("o").rangeBetween(-1, 0),
        "rows_to_end": ordered.rowsBetween(w.currentRow, w.unboundedFollowing),
        "rows_ahead": ordered.rowsBetween(1, 2),
    }


# `crates/krishiv-udf-pool/tests/window_iter_goldens.rs` (`UDFS`) repeats the bodies of `_window_udfs` and
# `_iter_udfs` to replay the goldens on the worker pool: change both together.
def _window_udfs(api: Api) -> dict[str, Any]:
    F = api.F

    @F.pandas_udf("double")
    def pandas_sum(s: pd.Series) -> float:
        return float(s.sum())

    @F.pandas_udf("double")
    def pandas_mean(s: pd.Series) -> float:
        return float(s.mean())

    @F.pandas_udf("long")
    def pandas_count(s: pd.Series) -> int:
        return len(s)

    @F.arrow_udf("double")
    def arrow_sum(a: pa.Array) -> float | None:
        return pc.call_function("sum", [a]).as_py()

    @F.arrow_udf("long")
    def arrow_count(a: pa.Array) -> int:
        return len(a)

    return {
        "pandas_sum": pandas_sum,
        "pandas_mean": pandas_mean,
        "pandas_count": pandas_count,
        "arrow_sum": arrow_sum,
        "arrow_count": arrow_count,
    }


def _window_case(api: Api, spark: Any, columns: list[tuple[str, str, str]]) -> Any:
    """``columns``: (output name, UDF name, window spec name); the UDFs take ``v``."""
    specs = _window_specs(api)
    udfs = _window_udfs(api)
    df = spark.createDataFrame(WINDOW_ROWS, WINDOW_SCHEMA)
    out = [udfs[u](df["v"]).over(specs[s]).alias(name) for name, u, s in columns]
    return df.select("id", *out).orderBy("id")


def _iter_udfs(api: Api) -> dict[str, Any]:
    F = api.F

    @F.pandas_udf("long")
    def pandas_plus_one(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for s in it:
            yield s + 1

    @F.pandas_udf("long")
    def pandas_rebatched(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        whole = pd.concat(list(it), ignore_index=True)
        yield whole.iloc[:3] * 2
        yield whole.iloc[3:] * 2

    @F.pandas_udf("long")
    def pandas_twice(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for s in it:
            yield s
            yield s

    @F.pandas_udf("long")
    def pandas_first_only(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        yield next(it)

    @F.pandas_udf("long")
    def pandas_short(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for s in it:
            yield s.iloc[:1]

    @F.pandas_udf("long")
    def pandas_not_iterator(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        return pd.Series([1, 2])  # type: ignore[return-value]  # ty: ignore[invalid-return-type]

    @F.arrow_udf("long")
    def arrow_plus_one(it: Iterator[pa.Array]) -> Iterator[pa.Array]:
        for a in it:
            yield pc.call_function("add", [a, 1])

    @F.arrow_udf("long")
    def arrow_twice(it: Iterator[pa.Array]) -> Iterator[pa.Array]:
        for a in it:
            yield a
            yield a

    @F.arrow_udf("long")
    def arrow_first_only(it: Iterator[pa.Array]) -> Iterator[pa.Array]:
        yield next(it)

    @F.arrow_udf("long")
    def arrow_short(it: Iterator[pa.Array]) -> Iterator[pa.Array]:
        for a in it:
            yield a.slice(0, 1)

    return {
        "pandas_plus_one": pandas_plus_one,
        "pandas_rebatched": pandas_rebatched,
        "pandas_twice": pandas_twice,
        "pandas_first_only": pandas_first_only,
        "pandas_short": pandas_short,
        "pandas_not_iterator": pandas_not_iterator,
        "arrow_plus_one": arrow_plus_one,
        "arrow_twice": arrow_twice,
        "arrow_first_only": arrow_first_only,
        "arrow_short": arrow_short,
    }


def _iter_case(api: Api, spark: Any, udf: str) -> Any:
    df = spark.createDataFrame(ITER_ROWS, ITER_SCHEMA).coalesce(1)
    return df.select("id", _iter_udfs(api)[udf](df["x"]).alias("r")).orderBy("id")


def _cases() -> dict[str, Callable[[Api, Any], Any]]:
    cases: dict[str, Callable[[Api, Any], Any]] = {}
    for spec in (
        "partition",
        "ordered_default",
        "rows_unbounded",
        "rows_sliding",
        "range_back",
        "rows_to_end",
        "rows_ahead",
    ):
        for kind, udfs in (("pandas", ("sum", "mean", "count")), ("arrow", ("sum", "count"))):
            cols = [(u, f"{kind}_{u}", spec) for u in udfs]
            cases[f"window/{kind}/{spec}"] = lambda api, spark, cols=cols: _window_case(api, spark, cols)
    # Unbounded and bounded frames in one projection, one eval type each: PySpark 4.2.0 refuses pandas and Arrow
    # window UDFs in one projection at planning (`INTERNAL_ERROR`, "All window functions must have the same eval
    # type in ArrowWindowPythonExec"), a Spark bug no golden pins.
    for kind, agg in (("pandas", "sum"), ("arrow", "count")):
        cols = [
            ("whole", f"{kind}_{agg}", "partition"),
            ("sliding", f"{kind}_{agg}", "rows_sliding"),
            ("unbounded", f"{kind}_count", "rows_unbounded"),
            ("ahead", f"{kind}_count", "rows_ahead"),
        ]
        cases[f"window/{kind}/mixed_bounds"] = lambda api, spark, cols=cols: _window_case(api, spark, cols)
    for udf in (
        "pandas_plus_one",
        "pandas_rebatched",
        "pandas_twice",
        "pandas_first_only",
        "pandas_short",
        "pandas_not_iterator",
        "arrow_plus_one",
        "arrow_twice",
        "arrow_first_only",
        "arrow_short",
    ):
        cases[f"iter/{udf}"] = lambda api, spark, udf=udf: _iter_case(api, spark, udf)
    return cases


CASES = _cases()


def cell(value: Any) -> Any:
    """A result cell as JSON: NaN and infinities as strings, everything else as is."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value
