# SPDX-License-Identifier: Apache-2.0
"""``mapInPandas``, ``mapInArrow``, ``applyInPandas``, ``applyInArrow`` and ``cogroup``, run in the driver.

PySpark runs these functions in executor Python workers, one call per partition or group. The embedded
engine has no such operator, so the functions run here, **when the method is called** (not at the next
action): the input is collected, the function is applied to Arrow batches (or per group), and the output
becomes a new in-memory :class:`DataFrame`. Results match PySpark's for functions that depend only on their
input; the memory needed is the whole input and output, and the order of groups is the order their keys
first appear.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from krishiv.errors import PySparkRuntimeError, PySparkTypeError, not_expected_type, not_implemented
from krishiv.sql import _convert

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame
    from krishiv.sql.types import StructType

_BATCH_ROWS_KEY = "spark.sql.execution.arrow.maxRecordsPerBatch"


def _struct(df: DataFrame, schema: Any) -> StructType:
    from krishiv.sql.types import StructType

    if isinstance(schema, StructType | str):
        out = df._session._explicit_schema(schema)
        return out
    raise not_expected_type("schema", "StructType or str", schema)


def _batch_rows(df: DataFrame) -> int:
    raw = df._session.conf.get(_BATCH_ROWS_KEY, "10000") or "10000"
    try:
        return max(int(raw), 1)
    except ValueError:
        return 10000


def _pandas() -> Any:
    try:
        import pandas
    except ImportError as e:
        raise ImportError("this method needs pandas; install it with `pip install krishiv[pandas]`") from e
    return pandas


def _to_pandas(df: DataFrame, table: Any) -> Any:
    _pandas()
    return _convert.session_local(table, df._session_zone()).to_pandas()


def _result_table(df: DataFrame, struct: StructType, parts: list[Any]) -> Any:
    """The tables ``parts`` (Arrow tables, one per function result) as one table of ``struct``."""
    from krishiv.sql.session import _pa_schema

    pa = _convert.require_pyarrow("this method")
    target = _pa_schema(struct, pa)
    aligned = [_align(p, struct, target) for p in parts]
    return pa.concat_tables(aligned) if aligned else target.empty_table()


def _align(table: Any, struct: StructType, target: Any) -> Any:
    """``table`` with the columns of ``target``: by name when every name matches, else by position."""
    names = [f.name for f in struct.fields]
    if table.num_columns != len(names):
        raise PySparkRuntimeError(
            f"Number of columns of the returned table doesn't match specified schema. "
            f"Expected: {len(names)} Actual: {table.num_columns}"
        )
    if set(table.schema.names) == set(names) and len(set(names)) == len(names):
        table = table.select(names)
    else:
        table = table.rename_columns(names)
    return table.cast(target)


def _wrap(df: DataFrame, struct: StructType, table: Any) -> DataFrame:
    from krishiv.sql.dataframe import DataFrame

    return DataFrame(df._session._native.create_dataframe(table), df._session)


def _from_pandas(pdf: Any) -> Any:
    pa = _convert.require_pyarrow("this method")
    if not isinstance(pdf, _pandas().DataFrame):
        raise PySparkTypeError(
            errorClass="UDF_RETURN_TYPE",
            messageParameters={"expected": "iterator of pandas.DataFrame", "actual": type(pdf).__name__},
        )
    return pa.Table.from_pandas(pdf, preserve_index=False)


def map_in_pandas(df: DataFrame, func: Callable[..., Any], schema: Any) -> DataFrame:
    """``func`` over an iterator of ``pandas.DataFrame`` batches; its yielded frames are the result."""
    struct = _struct(df, schema)
    table = df._arrow()
    pa = _convert.require_pyarrow("mapInPandas")
    batches: Iterator[Any] = (
        _to_pandas(df, pa.Table.from_batches([b])) for b in table.to_batches(max_chunksize=_batch_rows(df))
    )
    out = [_from_pandas(p) for p in func(batches)]
    return _wrap(df, struct, _result_table(df, struct, out))


def map_in_arrow(df: DataFrame, func: Callable[..., Any], schema: Any) -> DataFrame:
    """``func`` over an iterator of ``pyarrow.RecordBatch``; its yielded batches are the result."""
    pa = _convert.require_pyarrow("mapInArrow")
    struct = _struct(df, schema)
    table = df._arrow()
    out = []
    for b in func(iter(table.to_batches(max_chunksize=_batch_rows(df)))):
        if not isinstance(b, pa.RecordBatch):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={"expected": "iterator of pyarrow.RecordBatch", "actual": type(b).__name__},
            )
        out.append(pa.Table.from_batches([b]))
    return _wrap(df, struct, _result_table(df, struct, out))


def _arity(func: Callable[..., Any], allowed: tuple[int, ...]) -> int:
    try:
        n = len(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        n = allowed[0]
    if n not in allowed:
        raise PySparkTypeError(
            errorClass="INVALID_UDF_EVAL_TYPE", messageParameters={"eval_type": f"a function of {allowed} arguments"}
        )
    return n


def _groups(table: Any, keys: list[str]) -> dict[tuple[Any, ...], Any]:
    """``table`` split by the values of the columns ``keys``, in order of first appearance."""
    if not keys:
        return {(): table}
    indices: dict[tuple[Any, ...], list[int]] = {}
    key_rows = table.select(keys).to_pylist()
    for i, row in enumerate(key_rows):
        indices.setdefault(tuple(row[k] for k in keys), []).append(i)
    return {k: table.take(v) for k, v in indices.items()}


def _check_keys(keys: list[Any]) -> list[str]:
    for k in keys:
        if not isinstance(k, str):
            raise not_implemented("grouped map functions over a grouping expression (group by column names)")
    return list(keys)


def apply_in_pandas(df: DataFrame, keys: list[Any], func: Callable[..., Any], schema: Any) -> DataFrame:
    """``func(pdf)`` or ``func(key, pdf)`` per group of ``keys`` (column names)."""
    names = _check_keys(keys)
    struct = _struct(df, schema)
    arity = _arity(func, (1, 2))
    out = []
    for key, part in _groups(df._arrow(), names).items():
        pdf = _to_pandas(df, part)
        out.append(_from_pandas(func(pdf) if arity == 1 else func(key, pdf)))
    return _wrap(df, struct, _result_table(df, struct, out))


def apply_in_arrow(df: DataFrame, keys: list[Any], func: Callable[..., Any], schema: Any) -> DataFrame:
    """``func(table)`` or ``func(key, table)`` per group of ``keys`` (column names)."""
    pa = _convert.require_pyarrow("applyInArrow")
    names = _check_keys(keys)
    struct = _struct(df, schema)
    arity = _arity(func, (1, 2))
    out = []
    for key, part in _groups(df._arrow(), names).items():
        res = func(part) if arity == 1 else func(key, part)
        if not isinstance(res, pa.Table):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={"expected": "pyarrow.Table", "actual": type(res).__name__},
            )
        out.append(res)
    return _wrap(df, struct, _result_table(df, struct, out))


def cogroup_apply(
    left: tuple[DataFrame, list[Any]],
    right: tuple[DataFrame, list[Any]],
    func: Callable[..., Any],
    schema: Any,
    arrow: bool,
) -> DataFrame:
    """``func(l, r)`` or ``func(key, l, r)`` per key present in either side; a side without the key is empty."""
    pa = _convert.require_pyarrow("cogroup")
    ldf, rdf = left[0], right[0]
    lkeys, rkeys = _check_keys(left[1]), _check_keys(right[1])
    if len(lkeys) != len(rkeys):
        raise PySparkRuntimeError("Cogroup needs the same number of grouping keys on both sides.")
    struct = _struct(ldf, schema)
    arity = _arity(func, (2, 3))
    ltable, rtable = ldf._arrow(), rdf._arrow()
    lg, rg = _groups(ltable, lkeys), _groups(rtable, rkeys)
    ordered: list[tuple[Any, ...]] = list(lg) + [k for k in rg if k not in lg]
    out = []
    for key in ordered:
        lpart = lg.get(key, ltable.slice(0, 0))
        rpart = rg.get(key, rtable.slice(0, 0))
        if arrow:
            args: list[Any] = [lpart, rpart]
        else:
            args = [_to_pandas(ldf, lpart), _to_pandas(rdf, rpart)]
        res = func(*args) if arity == 2 else func(key, *args)
        out.append(res if arrow else _from_pandas(res))
    if arrow:
        for res in out:
            if not isinstance(res, pa.Table):
                raise PySparkTypeError(
                    errorClass="UDF_RETURN_TYPE",
                    messageParameters={"expected": "pyarrow.Table", "actual": type(res).__name__},
                )
    return _wrap(ldf, struct, _result_table(ldf, struct, out))

