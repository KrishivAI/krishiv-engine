# SPDX-License-Identifier: Apache-2.0
# Parts ported from Apache Spark v4.2.0, python/pyspark/worker.py and python/pyspark/sql/pandas/serializers.py
# (Apache License 2.0; see NOTICE). Each ported function names its source lines; the code keeps Spark's shape
# so a later diff against the tag stays readable.
"""The relation kinds of Python UDFs (design ``P3-T085-embedded-python-udf-kinds.md`` §3.2): map, grouped map,
cogrouped map, grouped aggregate, window aggregate and scalar iterator, each run over a stream of **frames**.

A window aggregate's partitions are groups; its runner yields one value per row of each, as Spark's JVM joins
them with the rows it queued (``ArrowWindowPythonEvaluatorFactory``). A bounded frame's frames lead with each
row's frame start and end (``INT`` row indexes within the partition), and the runner conf's
``window_bound_types`` says which (Spark's entry, one UDF per task here). A scalar iterator's frames are its
argument columns, a partition's batches then the end; it yields one value per input row in any batches.

A runner (:func:`make_relation`) is called once per task with a frame source and the task's key count, and
yields result batches in the engine's Arrow forms (strings and binaries as views, timestamps in UTC, as
``worker.py``'s scalar wrappers return them). The source's ``next()`` returns one of:

- ``("batch", side, batch)``: rows of the current group (of the partition, for a map kind); ``side`` is 0, or 1
  for a cogroup's right relation; a grouped kind's frames lead with its key columns, then the data columns;
- ``("end_group",)``: the group has no more rows;
- ``("end",)``: the partition has no more rows.

Spark's worker reads the JVM's grouped Arrow stream (``ArrowStreamGroupSerializer``: a count per group, then
the group's batches; the map kinds wrap each row in a struct). Here the engine's frames carry the same rows,
flat; each runner below is Spark's per-eval-type loop of ``read_udfs`` (``worker.py`` L2352-3560) between the
serializer's load and dump, with the struct wrapping the JVM adds and removes done in place where it changes
what the UDF sees (``mapInPandas`` builds its DataFrame from the struct, ``df_for_struct``). The engine sends
strings and binaries as Arrow views and zoned timestamps in UTC: they are cast to Spark's forms and the
session time zone before the conversion (``_to_spark``), and results back after it (:func:`_engine_batch`).
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from krishiv._udf._host import PySparkRuntimeError, PySparkTypeError, PySparkValueError
from krishiv._udf.types import StructField, StructType
from krishiv._udf.worker import (
    RunnerConf,
    _engine_arrow_type,
    _runner_conf,
    _spark_arrow_type,
    _to_spark,
    fail_on_stopiteration,
)

#: Spark's ``PythonEvalType`` codes (``pyspark/util.py:655-688``) of the relation kinds run here.
SQL_GROUPED_MAP_PANDAS_UDF = 201
SQL_GROUPED_AGG_PANDAS_UDF = 202
SQL_WINDOW_AGG_PANDAS_UDF = 203
SQL_SCALAR_PANDAS_ITER_UDF = 204
SQL_MAP_PANDAS_ITER_UDF = 205
SQL_COGROUPED_MAP_PANDAS_UDF = 206
SQL_MAP_ARROW_ITER_UDF = 207
SQL_GROUPED_MAP_ARROW_UDF = 209
SQL_COGROUPED_MAP_ARROW_UDF = 210
SQL_GROUPED_MAP_ARROW_ITER_UDF = 215
SQL_GROUPED_MAP_PANDAS_ITER_UDF = 216
SQL_GROUPED_AGG_PANDAS_ITER_UDF = 217
SQL_SCALAR_ARROW_ITER_UDF = 251
SQL_GROUPED_AGG_ARROW_UDF = 252
SQL_WINDOW_AGG_ARROW_UDF = 253
SQL_GROUPED_AGG_ARROW_ITER_UDF = 254

#: Every relation kind :func:`make_relation` runs.
RELATION_EVAL_TYPES = frozenset(
    {
        SQL_GROUPED_MAP_PANDAS_UDF,
        SQL_GROUPED_AGG_PANDAS_UDF,
        SQL_MAP_PANDAS_ITER_UDF,
        SQL_COGROUPED_MAP_PANDAS_UDF,
        SQL_MAP_ARROW_ITER_UDF,
        SQL_GROUPED_MAP_ARROW_UDF,
        SQL_COGROUPED_MAP_ARROW_UDF,
        SQL_GROUPED_MAP_ARROW_ITER_UDF,
        SQL_GROUPED_MAP_PANDAS_ITER_UDF,
        SQL_GROUPED_AGG_PANDAS_ITER_UDF,
        SQL_GROUPED_AGG_ARROW_UDF,
        SQL_GROUPED_AGG_ARROW_ITER_UDF,
        SQL_WINDOW_AGG_PANDAS_UDF,
        SQL_WINDOW_AGG_ARROW_UDF,
        SQL_SCALAR_PANDAS_ITER_UDF,
        SQL_SCALAR_ARROW_ITER_UDF,
    }
)

#: The grouped and window aggregate kinds: one value per group (per row of a partition, for a window), sent
#: together before the next frame is asked for.
AGGREGATE_EVAL_TYPES = frozenset(
    {
        SQL_GROUPED_AGG_PANDAS_UDF,
        SQL_GROUPED_AGG_PANDAS_ITER_UDF,
        SQL_GROUPED_AGG_ARROW_UDF,
        SQL_GROUPED_AGG_ARROW_ITER_UDF,
        SQL_WINDOW_AGG_PANDAS_UDF,
        SQL_WINDOW_AGG_ARROW_UDF,
    }
)

#: The window kind a grouped-aggregate UDF runs as over a window (Spark's ``ArrowWindowPythonExec`` eval type).
WINDOW_OF = {SQL_GROUPED_AGG_PANDAS_UDF: SQL_WINDOW_AGG_PANDAS_UDF, SQL_GROUPED_AGG_ARROW_UDF: SQL_WINDOW_AGG_ARROW_UDF}


class FrameSource(Protocol):
    """Where a runner reads its frames from (see the module doc)."""

    def next(self) -> tuple[Any, ...]: ...


class FrameError(Exception):
    """The frames broke the relation protocol (an engine bug, not the UDF's)."""


#: A task's runner: the frames and the key count in, the result batches out.
Runner = Callable[[FrameSource, int, int], Iterator[Any]]


# --- the frames ---
def _map_batches(source: FrameSource) -> Iterator[Any]:
    """A map task's batches, up to the end."""
    while True:
        frame = source.next()
        if frame[0] == "end":
            return
        if frame[0] != "batch":
            raise FrameError(f"a map task got {frame[0]!r}")
        yield frame[2]


class _Group:
    """One group's ``(side, batch)`` frames, up to its end marker; :meth:`drain` reads what the UDF left."""

    def __init__(self, source: FrameSource, first: tuple[int, Any]) -> None:
        self._source = source
        self._first: tuple[int, Any] | None = first
        self._ended = False

    def __iter__(self) -> _Group:
        return self

    def __next__(self) -> tuple[int, Any]:
        if self._first is not None:
            first, self._first = self._first, None
            return first
        if self._ended:
            raise StopIteration
        frame = self._source.next()
        if frame[0] == "batch":
            return frame[1], frame[2]
        if frame[0] == "end_group":
            self._ended = True
            raise StopIteration
        raise FrameError("the input ended inside a group")

    def drain(self) -> None:
        for _ in self:
            pass


def _groups(source: FrameSource) -> Iterator[_Group]:
    """A grouped task's groups, each read to its end before the next starts (as ``GroupPandasUDFSerializer``
    ``load_stream`` drains each group, ``serializers.py:681-690``)."""
    while True:
        frame = source.next()
        if frame[0] == "end":
            return
        if frame[0] != "batch":
            raise FrameError(f"a group starts with {frame[0]!r}")
        group = _Group(source, (frame[1], frame[2]))
        yield group
        group.drain()


def _spark_batch(batch: Any, timezone: str | None) -> Any:
    """``batch`` with Spark's Arrow forms: strings and binaries not as views, zoned timestamps in the session
    time zone (the same instants)."""
    import pyarrow as pa

    if batch.num_columns == 0:
        return batch
    return pa.RecordBatch.from_arrays([_to_spark(c, timezone) for c in batch.columns], names=batch.schema.names)


def _sides(group: _Group, timezone: str | None) -> tuple[list[Any], list[Any]]:
    """A cogroup's left and right batches (``ArrowStreamCoGroupSerializer.load_stream``,
    ``serializers.py:205-220``): every key has at least one frame per side, zero rows for a side without."""
    left: list[Any] = []
    right: list[Any] = []
    for side, batch in group:
        if side == 0 and not right:
            left.append(_spark_batch(batch, timezone))
        elif side == 1:
            right.append(_spark_batch(batch, timezone))
        else:
            raise FrameError(f"a cogroup frame of side {side} out of order")
    if not left or not right:
        raise FrameError("a cogroup key without a frame for each side")
    return left, right


# --- the results ---
def _engine_form(t: Any) -> Any:
    """The engine's Arrow form of a Spark Arrow type: views for strings and binaries, zoned timestamps in UTC."""
    return _engine_arrow_type(_spark_arrow_type(t, "UTC"))


def _engine_schema(return_type: Any, conf: RunnerConf) -> Any:
    """The engine's Arrow schema of the result: the struct's fields, or one ``_0`` for a scalar type."""
    import pyarrow as pa

    from krishiv._udf.pandas_types import to_arrow_type

    fields = return_type.fields if isinstance(return_type, StructType) else [StructField("_0", return_type)]
    return pa.schema(
        [
            pa.field(
                f.name,
                _engine_form(to_arrow_type(f.dataType, timezone="UTC", prefers_large_types=conf.use_large_var_types)),
            )
            for f in fields
        ]
    )


def _engine_batch(arrays: list[Any], schema: Any, num_rows: int, fields: Any = None) -> Any:
    """The result columns in the engine's forms. A column whose type differs from the declared one by more
    than its form (a ``mapInArrow`` result of another type, which Spark's worker does not check) is left as
    it is: the server's check of the result names the mismatch. ``fields`` are the UDF's own result fields
    (``mapInArrow``): their names and nullability are kept, as the JVM sees them, for the server's check under
    ``spark.sql.execution.arrow.pyspark.validateSchema.enabled`` (``MapInBatchEvaluatorFactory.scala``
    L81-91 at v4.2.0); otherwise the declared names are used."""
    import pyarrow as pa

    if not arrays:
        return pa.RecordBatch.from_pydict({"_": [None] * num_rows}).select([])
    out = []
    for arr, field in zip(arrays, schema, strict=False):
        if arr.type != field.type and _engine_form(arr.type) == field.type:
            arr = arr.cast(field.type)
        out.append(arr)
    if fields is not None:
        # Every column the UDF returned, so the server's checks see the extra ones (review R2-4).
        out.extend(arrays[len(out) :])
        own = pa.schema([pa.field(f.name, arr.type, nullable=f.nullable) for f, arr in zip(fields, out, strict=True)])
        return pa.RecordBatch.from_arrays(out, schema=own)
    names = [f.name for f in schema][: len(out)]
    return pa.RecordBatch.from_arrays(out, names=names)


def _from_pandas(result: Any, return_type: Any, conf: RunnerConf, schema: Any) -> Any:
    """``ArrowStreamPandasUDFSerializer.dump_stream``'s ``create_batch`` (``serializers.py:509-539``) for one
    ``(result, return_type)``: a struct column for a struct type, which the JVM takes apart
    (``MapInBatchEvaluatorFactory``, ``FlatMapGroupsInBatchExec``)."""
    import pandas as pd

    from krishiv._udf.conversion import PandasToArrowConversion

    if isinstance(return_type, StructType) and not isinstance(result, pd.DataFrame):
        raise PySparkValueError(
            "Invalid return type. Please make sure that the UDF returns a "
            "pandas.DataFrame when the specified return type is StructType."
        )
    column = PandasToArrowConversion.convert(
        [result],
        StructType([StructField("_0", return_type)]),
        timezone=conf.timezone,
        safecheck=conf.safecheck,
        arrow_cast=True,
        prefers_large_types=conf.use_large_var_types,
        assign_cols_by_name=conf.assign_cols_by_name,
        int_to_decimal_coercion_enabled=conf.int_to_decimal_coercion_enabled,
        ignore_unexpected_complex_type_values=False,
        is_legacy=False,
    ).column(0)
    arrays = column.flatten() if isinstance(return_type, StructType) else [column]
    return _engine_batch(arrays, schema, len(column))


def _from_arrow(batch: Any, schema: Any, own_fields: bool = False) -> Any:
    fields = list(batch.schema) if own_fields else None
    return _engine_batch(list(batch.columns), schema, batch.num_rows, fields=fields)


# --- ported from pyspark/worker.py:237-276 ---
def verify_result(expected_type: type) -> Callable[[Any], Iterator]:
    """
    Create a result verifier that checks both iterability and element types.
    """
    package = getattr(inspect.getmodule(expected_type), "__package__", "")
    label: str = f"{package}.{expected_type.__name__}"

    def check_element(element: Any) -> Any:
        if not isinstance(element, expected_type):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={
                    "expected": f"iterator of {label}",
                    "actual": f"iterator of {type(element).__name__}",
                },
            )
        return element

    def check(result: Any) -> Iterator:
        if not isinstance(result, Iterator) and not hasattr(result, "__iter__"):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={
                    "expected": f"iterator of {label}",
                    "actual": type(result).__name__,
                },
            )
        return map(check_element, result)

    return check


# --- ported from pyspark/worker.py:417-451 ---
def wrap_pandas_batch_iter_udf(f: Callable[..., Any], return_type: Any) -> Callable[..., Any]:
    iter_type_label = "pandas.DataFrame" if isinstance(return_type, StructType) else "pandas.Series"

    def verify_result(result: Any) -> Any:
        if not isinstance(result, Iterator) and not hasattr(result, "__iter__"):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={
                    "expected": f"iterator of {iter_type_label}",
                    "actual": type(result).__name__,
                },
            )
        return result

    def verify_element(elem: Any) -> Any:
        import pandas as pd

        if not isinstance(elem, pd.DataFrame if isinstance(return_type, StructType) else pd.Series):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={
                    "expected": f"iterator of {iter_type_label}",
                    "actual": f"iterator of {type(elem).__name__}",
                },
            )

        verify_pandas_result(elem, return_type, assign_cols_by_name=True, truncate_return_schema=True)

        return elem

    return lambda *iterator: map(lambda res: (res, return_type), map(verify_element, verify_result(f(*iterator))))


# --- ported from pyspark/worker.py:454-511 ---
def verify_pandas_result(
    result: Any, return_type: Any, assign_cols_by_name: bool, truncate_return_schema: bool
) -> None:
    import pandas as pd

    if isinstance(return_type, StructType):
        if not isinstance(result, pd.DataFrame):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={
                    "expected": "pandas.DataFrame",
                    "actual": type(result).__name__,
                },
            )

        # check the schema of the result only if it is not empty or has columns
        if not result.empty or len(result.columns) != 0:
            # if any column name of the result is a string
            # the column names of the result have to match the return type
            #   see create_array in pyspark.sql.pandas.serializers.ArrowStreamPandasSerializer
            field_names = {field.name for field in return_type.fields}
            # only the first len(field_names) result columns are considered
            # when truncating the return schema
            result_columns = result.columns[: len(field_names)] if truncate_return_schema else result.columns
            column_names = set(result_columns)
            if (
                assign_cols_by_name
                and any(isinstance(name, str) for name in result.columns)
                and column_names != field_names
            ):
                missing_names = sorted(field_names.difference(column_names))
                missing = f" Missing: {', '.join(missing_names)}." if missing_names else ""

                extra_names = sorted(column_names.difference(field_names))
                extra = f" Unexpected: {', '.join(extra_names)}." if extra_names else ""

                raise PySparkRuntimeError(
                    errorClass="RESULT_COLUMN_NAMES_MISMATCH",
                    messageParameters={
                        "missing": missing,
                        "extra": extra,
                    },
                )
            # otherwise the number of columns of result have to match the return type
            elif len(result_columns) != len(return_type):
                raise PySparkRuntimeError(
                    errorClass="RESULT_COLUMN_SCHEMA_MISMATCH",
                    messageParameters={
                        "expected": str(len(return_type)),
                        "actual": str(len(result.columns)),
                    },
                )
    else:
        if not isinstance(result, pd.Series):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={"expected": "pandas.Series", "actual": type(result).__name__},
            )


# --- ported from pyspark/worker.py:564-652 ---
def verify_arrow_result(result: Any, assign_cols_by_name: bool, expected_cols_and_types: Any) -> None:
    # the types of the fields have to be identical to return type
    # an empty table can have no columns; if there are columns, they have to match
    if result.num_columns != 0 or result.num_rows != 0:
        # columns are either mapped by name or position
        if assign_cols_by_name:
            actual_cols_and_types = dict(zip(result.schema.names, result.schema.types, strict=False))
            missing_names = sorted(set(expected_cols_and_types.keys()).difference(actual_cols_and_types.keys()))
            extra_names = sorted(set(actual_cols_and_types.keys()).difference(expected_cols_and_types.keys()))

            if missing_names or extra_names:
                missing = f" Missing: {', '.join(missing_names)}." if missing_names else ""
                extra = f" Unexpected: {', '.join(extra_names)}." if extra_names else ""

                raise PySparkRuntimeError(
                    errorClass="RESULT_COLUMN_NAMES_MISMATCH",
                    messageParameters={
                        "missing": missing,
                        "extra": extra,
                    },
                )

            column_types = [
                (name, expected_cols_and_types[name], actual_cols_and_types[name])
                for name in sorted(expected_cols_and_types.keys())
            ]
        else:
            actual_cols_and_types = list(zip(result.schema.names, result.schema.types, strict=False))
            column_types = [
                (expected_name, expected_type, actual_type)
                for (expected_name, expected_type), (_actual_name, actual_type) in zip(
                    expected_cols_and_types, actual_cols_and_types, strict=False
                )
            ]

        type_mismatch = [(name, expected, actual) for name, expected, actual in column_types if actual != expected]

        if type_mismatch:
            raise PySparkRuntimeError(
                errorClass="RESULT_COLUMN_TYPES_MISMATCH",
                messageParameters={
                    "mismatch": ", ".join(
                        f"column '{name}' (expected {expected}, actual {actual})"
                        for name, expected, actual in type_mismatch
                    )
                },
            )


def verify_arrow_table(table: Any, assign_cols_by_name: bool, expected_cols_and_types: Any) -> None:
    import pyarrow as pa

    if not isinstance(table, pa.Table):
        raise PySparkTypeError(
            errorClass="UDF_RETURN_TYPE",
            messageParameters={
                "expected": "pyarrow.Table",
                "actual": type(table).__name__,
            },
        )

    verify_arrow_result(table, assign_cols_by_name, expected_cols_and_types)


def verify_arrow_batch(batch: Any, assign_cols_by_name: bool, expected_cols_and_types: Any) -> None:
    import pyarrow as pa

    if not isinstance(batch, pa.RecordBatch):
        raise PySparkTypeError(
            errorClass="UDF_RETURN_TYPE",
            messageParameters={
                "expected": "pyarrow.RecordBatch",
                "actual": type(batch).__name__,
            },
        )

    verify_arrow_result(batch, assign_cols_by_name, expected_cols_and_types)


def _expected_cols_and_types(return_type: Any, conf: RunnerConf) -> Any:
    """``worker.py:2797-2805``: the declared columns by name, or by position without
    ``assignColumnsByName``."""
    from krishiv._udf.pandas_types import to_arrow_type

    if conf.assign_cols_by_name:
        return {col.name: to_arrow_type(col.dataType, timezone="UTC") for col in return_type.fields}
    return [(col.name, to_arrow_type(col.dataType, timezone="UTC")) for col in return_type.fields]


def _reordered(batch: Any, arrow_return_type: Any, conf: RunnerConf) -> Any:
    """The result's columns in the declared order when they are bound by name (``worker.py:2847-2852``)."""
    import pyarrow as pa

    if not conf.assign_cols_by_name:
        return batch
    return pa.RecordBatch.from_arrays(
        [batch.column(field.name) for field in arrow_return_type],
        names=[field.name for field in arrow_return_type],
    )


def _to_pandas(batch: Any, conf: RunnerConf, **options: Any) -> list[Any]:
    from krishiv._udf.conversion import ArrowBatchTransformer

    return ArrowBatchTransformer.to_pandas(
        batch,
        timezone=conf.timezone,  # ty: ignore[invalid-argument-type]  # Optional, as Spark passes it
        prefer_int_ext_dtype=conf.prefer_int_ext_dtype,
        **options,
    )


# --- SQL_MAP_PANDAS_ITER_UDF (205): worker.py:3184-3246 with ArrowStreamPandasUDFSerializer(df_for_struct) ---
def _map_pandas(f: Callable[..., Any], return_type: Any) -> Runner:
    from krishiv._udf.conversion import ArrowBatchTransformer

    udf = wrap_pandas_batch_iter_udf(fail_on_stopiteration(f), return_type)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)

        def frames() -> Iterator[Any]:
            for batch in _map_batches(source):
                # The JVM wraps each row in a struct (`MapInBatchEvaluatorFactory`), which load_stream turns into
                # one DataFrame (`df_for_struct`, `serializers.py:443-455`).
                if batch.num_columns == 0:
                    # No field to concatenate: a frame of the rows alone (the conversion has nothing to build it from).
                    import pandas as pd

                    yield pd.DataFrame(index=range(batch.num_rows))
                    continue
                struct = ArrowBatchTransformer.wrap_struct(_spark_batch(batch, conf.timezone))
                yield _to_pandas(struct, conf, struct_in_pandas="dict", ndarray_as_list=False, df_for_struct=True)[0]

        for result, spark_type in udf(frames()):
            yield _from_pandas(result, spark_type, conf, schema)

    return run


# --- SQL_MAP_ARROW_ITER_UDF (207): worker.py:2529-2552 ---
def _map_arrow(f: Callable[..., Any], return_type: Any) -> Runner:
    import pyarrow as pa

    udf_func = fail_on_stopiteration(f)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        # flatten_struct: the engine's frames are flat already.
        input_batches = (_spark_batch(b, conf.timezone) for b in _map_batches(source))
        output_batches = udf_func(input_batches)
        for batch in verify_result(pa.RecordBatch)(output_batches):
            yield _from_arrow(batch, schema, own_fields=True)

    return run


# --- SQL_GROUPED_MAP_PANDAS_UDF (201): worker.py:2929-2997 ---
def _grouped_map_pandas(f: Callable[..., Any], return_type: Any) -> Runner:
    import pandas as pd
    import pyarrow as pa

    num_udf_args = len(inspect.getfullargspec(f).args)
    grouped_udf = fail_on_stopiteration(f)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        for group in _groups(source):
            all_batches = [_spark_batch(b, conf.timezone) for _side, b in group]
            table = pa.Table.from_batches(all_batches).combine_chunks()
            all_series = _to_pandas(table, conf)
            value_df = pd.concat(all_series[key_count:], axis=1)

            if num_udf_args == 1:
                result = grouped_udf(value_df)
            else:
                key = tuple(all_series[o].iloc[0] for o in range(key_count))
                result = grouped_udf(key, value_df)

            del all_batches, table, all_series, value_df

            verify_pandas_result(result, return_type, conf.assign_cols_by_name, truncate_return_schema=False)
            yield _from_pandas(result, return_type, conf, schema)
            del result

    return run


# --- SQL_GROUPED_MAP_PANDAS_ITER_UDF (216): worker.py:3248-3277 around wrap_grouped_map_pandas_iter_udf
# (worker.py:655-685) ---
def _grouped_map_pandas_iter(f: Callable[..., Any], return_type: Any) -> Runner:
    import pandas as pd

    argspec = inspect.getfullargspec(f)
    udf = fail_on_stopiteration(f)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        for group in _groups(source):
            batch_iter = (_spark_batch(b, conf.timezone) for _side, b in group)
            first_series = _to_pandas(next(batch_iter), conf)
            key_series = first_series[:key_count]

            def value_series_gen(first: list[Any] = first_series, rest: Iterator[Any] = batch_iter) -> Iterator[Any]:
                yield first[key_count:]
                for batch in rest:
                    yield _to_pandas(batch, conf)[key_count:]

            def dataframe_iter(values: Iterator[Any]) -> Iterator[Any]:
                for value_series in values:
                    yield pd.concat(value_series, axis=1)

            if len(argspec.args) == 1:
                result = udf(dataframe_iter(value_series_gen()))
            elif len(argspec.args) == 2:
                key = tuple(s.iloc[0] for s in key_series)
                result = udf(key, dataframe_iter(value_series_gen()))
            else:
                raise UnboundLocalError("cannot access local variable 'result' where it is not associated with a value")

            for df in result:
                verify_pandas_result(df, return_type, conf.assign_cols_by_name, truncate_return_schema=False)
                yield _from_pandas(df, return_type, conf, schema)

    return run


# --- SQL_GROUPED_MAP_ARROW_UDF (209) and SQL_GROUPED_MAP_ARROW_ITER_UDF (215): worker.py:2787-2927 ---
def _grouped_map_arrow(f: Callable[..., Any], return_type: Any, iterator: bool) -> Runner:
    import pyarrow as pa

    from krishiv._udf.pandas_types import to_arrow_type

    num_udf_args = len(inspect.getfullargspec(f).args)
    grouped_udf = fail_on_stopiteration(f)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        arrow_return_type = to_arrow_type(return_type, timezone="UTC", prefers_large_types=conf.use_large_var_types)
        expected_cols_and_types = _expected_cols_and_types(return_type, conf)
        for group in _groups(source):
            batches = (_spark_batch(b, conf.timezone) for _side, b in group)
            first_batch = next(batches)
            key_offsets = range(key_count)
            value_offsets = range(key_count, first_batch.num_columns)
            keys = pa.RecordBatch.from_arrays(
                [first_batch.columns[o] for o in key_offsets],
                [first_batch.schema.names[o] for o in key_offsets],
            )
            value_batches = (
                pa.RecordBatch.from_arrays(
                    [b.columns[o] for o in value_offsets],
                    [b.schema.names[o] for o in value_offsets],
                )
                for b in itertools.chain((first_batch,), batches)
            )
            if not iterator:
                value_table = pa.Table.from_batches(value_batches)
                if num_udf_args == 1:
                    result = grouped_udf(value_table)
                else:
                    key = tuple(c[0] for c in keys.columns)
                    result = grouped_udf(key, value_table)

                verify_arrow_table(result, conf.assign_cols_by_name, expected_cols_and_types)

                for batch in result.to_batches():
                    yield _from_arrow(_reordered(batch, arrow_return_type, conf), schema)
            else:
                if num_udf_args == 1:
                    result = grouped_udf(value_batches)
                else:
                    key = tuple(c[0] for c in keys.columns)
                    result = grouped_udf(key, value_batches)

                for batch in result:
                    verify_arrow_batch(batch, conf.assign_cols_by_name, expected_cols_and_types)
                    yield _from_arrow(_reordered(batch, arrow_return_type, conf), schema)

                # Drain remaining input batches to maintain stream position
                for _ in value_batches:
                    pass

    return run


# --- SQL_COGROUPED_MAP_PANDAS_UDF (206): worker.py:3445-3458 around wrap_cogrouped_map_pandas_udf
# (worker.py:542-561) and CogroupPandasUDFSerializer.load_stream (serializers.py:725-755) ---
def _cogrouped_map_pandas(f: Callable[..., Any], return_type: Any) -> Runner:
    import pandas as pd
    import pyarrow as pa

    from krishiv._udf.pandas_types import from_arrow_schema

    argspec = inspect.getfullargspec(f)
    udf = fail_on_stopiteration(f)

    def wrapped(left_key_series: Any, left_value_series: Any, right_key_series: Any, right_value_series: Any) -> Any:
        left_df = pd.concat(left_value_series, axis=1)
        right_df = pd.concat(right_value_series, axis=1)

        if len(argspec.args) == 2:
            result = udf(left_df, right_df)
        elif len(argspec.args) == 3:
            key_series = left_key_series if not left_df.empty else right_key_series
            key = tuple(s[0] for s in key_series)
            result = udf(key, left_df, right_df)
        else:
            raise UnboundLocalError("cannot access local variable 'result' where it is not associated with a value")
        return result

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        for group in _groups(source):
            left, right = _sides(group, conf.timezone)
            series = []
            for batches in (left, right):
                table = pa.Table.from_batches(batches)
                series.append(
                    _to_pandas(
                        table,
                        conf,
                        schema=from_arrow_schema(table.schema),
                        struct_in_pandas="dict",
                        ndarray_as_list=False,
                        df_for_struct=False,
                    )
                )
            result = wrapped(series[0][:key_count], series[0][key_count:], series[1][:key_count], series[1][key_count:])
            verify_pandas_result(result, return_type, conf.assign_cols_by_name, truncate_return_schema=False)
            yield _from_pandas(result, return_type, conf, schema)

    return run


# --- SQL_COGROUPED_MAP_ARROW_UDF (210): worker.py:3460-3484 around wrap_cogrouped_map_arrow_udf
# (worker.py:514-539) and ArrowStreamGroupUDFSerializer.dump_stream (serializers.py:337-360) ---
def _cogrouped_map_arrow(f: Callable[..., Any], return_type: Any) -> Runner:
    import pyarrow as pa

    from krishiv._udf.pandas_types import to_arrow_type

    argspec = inspect.getfullargspec(f)
    udf = fail_on_stopiteration(f)

    def batch_from_offset(batch: Any, offsets: range) -> Any:
        return pa.RecordBatch.from_arrays(
            arrays=[batch.columns[o] for o in offsets],
            names=[batch.schema.names[o] for o in offsets],
        )

    def table_from_batches(batches: list[Any], offsets: range) -> Any:
        return pa.Table.from_batches([batch_from_offset(batch, offsets) for batch in batches])

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        expected_cols_and_types = _expected_cols_and_types(return_type, conf)
        arrow_type = to_arrow_type(return_type, timezone="UTC")
        for group in _groups(source):
            left, right = _sides(group, conf.timezone)
            keys = range(key_count)
            left_key_table = table_from_batches(left, keys)
            left_value_table = table_from_batches(left, range(key_count, left[0].num_columns))
            right_key_table = table_from_batches(right, keys)
            right_value_table = table_from_batches(right, range(key_count, right[0].num_columns))
            if len(argspec.args) == 2:
                result = udf(left_value_table, right_value_table)
            elif len(argspec.args) == 3:
                key_table = left_key_table if left_key_table.num_rows > 0 else right_key_table
                key = tuple(c[0] for c in key_table.columns)
                result = udf(key, left_value_table, right_value_table)
            else:
                raise UnboundLocalError("cannot access local variable 'result' where it is not associated with a value")

            verify_arrow_table(result, conf.assign_cols_by_name, expected_cols_and_types)

            for batch in result.to_batches():
                yield _from_arrow(_reordered(batch, arrow_type, conf), schema)

    return run


# --- SQL_GROUPED_AGG_PANDAS_UDF (202): worker.py:3509-3536 around wrap_grouped_agg_pandas_udf
# (worker.py:842-854) and ArrowStreamAggPandasUDFSerializer.load_stream (serializers.py:625-652); and its
# iterator form SQL_GROUPED_AGG_PANDAS_ITER_UDF (217): worker.py:3486-3507, wrap_grouped_agg_pandas_iter_udf
# (worker.py:857-872) ---
def _grouped_agg_pandas(f: Callable[..., Any], return_type: Any, iterator: bool) -> Runner:
    import pandas as pd

    udf = fail_on_stopiteration(f)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        arg_offsets = range(arity)
        for group in _groups(source):
            series_iter = (
                tuple(
                    _to_pandas(
                        _spark_batch(b, conf.timezone),
                        conf,
                        struct_in_pandas="dict",
                        ndarray_as_list=False,
                        df_for_struct=False,
                    )
                )
                for _side, b in group
            )
            if iterator:
                if len(arg_offsets) == 1:
                    args_iter: Iterator[Any] = (batch_series[arg_offsets[0]] for batch_series in series_iter)
                else:
                    args_iter = (tuple(batch_series[o] for o in arg_offsets) for batch_series in series_iter)
                result = udf(args_iter)
            else:
                batches = list(series_iter)
                num_columns = len(batches[0])
                concatenated = [
                    pd.concat([batch[i] for batch in batches], ignore_index=True) for i in range(num_columns)
                ]
                result = udf(*[concatenated[o] for o in arg_offsets])
            yield _from_pandas(pd.Series([result]), return_type, conf, schema)

    return run


# --- SQL_GROUPED_AGG_ARROW_UDF (252): worker.py:2648-2686; SQL_GROUPED_AGG_ARROW_ITER_UDF (254):
# worker.py:2688-2717 ---
def _grouped_agg_arrow(f: Callable[..., Any], return_type: Any, iterator: bool) -> Runner:
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowBatchTransformer
    from krishiv._udf.pandas_types import to_arrow_schema

    udf_func = fail_on_stopiteration(f)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        return_schema = to_arrow_schema(
            StructType([StructField("_0", return_type)]),
            timezone="UTC",
            prefers_large_types=conf.use_large_var_types,
        )
        args_offsets = range(arity)

        def extract_args(batch: Any) -> Any:
            args = tuple(batch.column(o) for o in args_offsets)
            return args[0] if len(args) == 1 else args

        for group in _groups(source):
            batches = (_spark_batch(b, conf.timezone) for _side, b in group)
            if iterator:
                batch_iter = map(extract_args, batches)
                result = udf_func(batch_iter)
                # Drain remaining batches to maintain stream position
                for _ in batch_iter:
                    pass
            else:
                batch_list = list(batches)
                if hasattr(pa, "concat_batches"):
                    concatenated = pa.concat_batches(batch_list)
                else:
                    # pyarrow.concat_batches not supported before 19.0.0
                    concatenated = pa.RecordBatch.from_struct_array(
                        pa.concat_arrays([b.to_struct_array() for b in batch_list])
                    )
                result = udf_func(*[concatenated.column(o) for o in args_offsets])
            batch = pa.RecordBatch.from_arrays([pa.array([result])], ["_0"])
            yield _from_arrow(ArrowBatchTransformer.enforce_schema(batch, return_schema), schema)

    return run


# --- the window kinds: worker.py:875-954 (wrap_window_agg_pandas_udf and its two forms) ---
def _window_bound_type(conf: RunnerConf) -> str:
    """The task's frame kind from the runner conf's ``window_bound_types`` (``worker.py:875-894``): one UDF per
    task, so the first entry."""
    window_bound_types_str = conf.get("window_bound_types")
    window_bound_type = window_bound_types_str.split(",")[0].strip().lower()
    if window_bound_type not in ("bounded", "unbounded"):
        raise PySparkRuntimeError(
            errorClass="INVALID_WINDOW_BOUND_TYPE",
            messageParameters={
                "window_bound_type": window_bound_type,
            },
        )
    return window_bound_type


def _concat_batches(batch_list: list[Any]) -> Any:
    """``pa.concat_batches`` (``worker.py:2735-2743``), with Spark's fallback before pyarrow 19."""
    import pyarrow as pa

    if hasattr(pa, "concat_batches"):
        return pa.concat_batches(batch_list)
    return pa.RecordBatch.from_struct_array(pa.concat_arrays([b.to_struct_array() for b in batch_list]))


# --- SQL_WINDOW_AGG_PANDAS_UDF (203): wrap_unbounded_window_agg_pandas_udf and
# wrap_bounded_window_agg_pandas_udf (worker.py:897-954) under read_udfs' mapper (worker.py:3509-3536) and
# ArrowStreamAggPandasUDFSerializer.load_stream (serializers.py:625-652) ---
def _window_agg_pandas(f: Callable[..., Any], return_type: Any) -> Runner:
    import pandas as pd

    func = fail_on_stopiteration(f)

    def unbounded(*series: Any) -> Any:
        result = func(*series)
        return pd.Series([result]).repeat(len(series[0]))

    def bounded(begin_index: Any, end_index: Any, *series: Any) -> Any:
        result = []
        begin_array = begin_index.values
        end_array = end_index.values
        for i in range(len(begin_array)):
            series_slices = [s.iloc[begin_array[i] : end_array[i]] for s in series]
            result.append(func(*series_slices))
        return pd.Series(result)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        wrapped = bounded if _window_bound_type(conf) == "bounded" else unbounded
        for group in _groups(source):
            batches = [
                tuple(
                    _to_pandas(
                        _spark_batch(b, conf.timezone),
                        conf,
                        struct_in_pandas="dict",
                        ndarray_as_list=False,
                        df_for_struct=False,
                    )
                )
                for _side, b in group
            ]
            num_columns = len(batches[0])
            concatenated = [pd.concat([batch[i] for batch in batches], ignore_index=True) for i in range(num_columns)]
            yield _from_pandas(wrapped(*concatenated), return_type, conf, schema)

    return run


# --- SQL_WINDOW_AGG_ARROW_UDF (253): worker.py:2719-2785 ---
def _window_agg_arrow(f: Callable[..., Any], return_type: Any) -> Runner:
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowBatchTransformer
    from krishiv._udf.pandas_types import to_arrow_schema

    udf_func = fail_on_stopiteration(f)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _engine_schema(return_type, conf)
        bound_type = _window_bound_type(conf)
        return_schema = to_arrow_schema(
            StructType([StructField("_0", return_type)]),
            timezone="UTC",
            prefers_large_types=conf.use_large_var_types,
        )
        for group in _groups(source):
            batch_list = [_spark_batch(b, conf.timezone) for _side, b in group]
            if not batch_list:
                continue
            concatenated = _concat_batches(batch_list)
            num_rows = concatenated.num_rows
            if bound_type == "unbounded":
                result = udf_func(*[concatenated.column(o) for o in range(concatenated.num_columns)])
                result_array = pa.repeat(result, num_rows)
            else:
                begin_col = concatenated.column(0)
                end_col = concatenated.column(1)
                results = []
                for i in range(num_rows):
                    offset = begin_col[i].as_py()
                    length = end_col[i].as_py() - offset
                    slices = [
                        concatenated.column(o).slice(offset=offset, length=length)
                        for o in range(2, concatenated.num_columns)
                    ]
                    results.append(udf_func(*slices))
                result_array = pa.array(results)
            batch = pa.RecordBatch.from_arrays([result_array], ["_0"])
            yield _from_arrow(ArrowBatchTransformer.enforce_schema(batch, return_schema), schema)

    return run


# --- the scalar iterator kinds ---
def _column_schema(return_type: Any, conf: RunnerConf) -> Any:
    """The engine's schema of a scalar kind's result: one column ``_0`` of the return type (a struct stays one
    column)."""
    import pyarrow as pa

    from krishiv._udf.pandas_types import to_arrow_type

    arrow_type = to_arrow_type(return_type, timezone="UTC", prefers_large_types=conf.use_large_var_types)
    return pa.schema([pa.field("_0", _engine_form(arrow_type))])


def _arg_batches(source: FrameSource, conf: RunnerConf) -> Iterator[Any]:
    """A scalar iterator task's argument batches, in Spark's Arrow forms."""
    for batch in _map_batches(source):
        yield _spark_batch(batch, conf.timezone)


# --- SQL_SCALAR_PANDAS_ITER_UDF (204): worker.py:3183-3246 around wrap_pandas_batch_iter_udf (worker.py:415-451),
# with ArrowStreamPandasUDFSerializer(df_for_struct=True) and its dump_stream's create_batch
# (serializers.py:509-539) ---
def _scalar_pandas_iter(f: Callable[..., Any], return_type: Any) -> Runner:
    import pandas as pd

    from krishiv._udf.conversion import PandasToArrowConversion

    # `read_single_udf` does not wrap this eval type in `fail_on_stopiteration` (worker.py:1103-1110).
    udf = wrap_pandas_batch_iter_udf(f, return_type)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        from krishiv._udf._host import PySparkValueError

        conf = _runner_conf()
        schema = _column_schema(return_type, conf)
        num_input_rows = 0

        def map_batch(batch: Any) -> Any:
            nonlocal num_input_rows

            udf_args = _to_pandas(batch, conf, struct_in_pandas="dict", ndarray_as_list=False, df_for_struct=True)
            num_input_rows += len(udf_args[0])
            if len(udf_args) == 1:
                return udf_args[0]
            return tuple(udf_args)

        iterator = map(map_batch, _arg_batches(source, conf))
        result_iter = udf(iterator)

        num_output_rows = 0
        for result_batch, result_type in result_iter:
            num_output_rows += len(result_batch)
            # This check is for Scalar Iterator UDF to fail fast.
            if num_output_rows > num_input_rows:
                raise PySparkRuntimeError(errorClass="OUTPUT_EXCEEDS_INPUT_ROWS", messageParameters={})
            if isinstance(result_type, StructType) and not isinstance(result_batch, pd.DataFrame):
                raise PySparkValueError(
                    "Invalid return type. Please make sure that the UDF returns a "
                    "pandas.DataFrame when the specified return type is StructType."
                )
            column = PandasToArrowConversion.convert(
                [result_batch],
                StructType([StructField("_0", result_type)]),
                timezone=conf.timezone,
                safecheck=conf.safecheck,
                arrow_cast=True,
                prefers_large_types=conf.use_large_var_types,
                assign_cols_by_name=conf.assign_cols_by_name,
                int_to_decimal_coercion_enabled=conf.int_to_decimal_coercion_enabled,
                ignore_unexpected_complex_type_values=False,
                is_legacy=False,
            ).column(0)
            yield _engine_batch([column], schema, len(column))

        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise PySparkRuntimeError(
                errorClass="INPUT_NOT_FULLY_CONSUMED",
                messageParameters={},
            )

        if num_output_rows != num_input_rows:
            raise PySparkRuntimeError(
                errorClass="RESULT_ROWS_MISMATCH",
                messageParameters={
                    "output_length": str(num_output_rows),
                    "input_length": str(num_input_rows),
                },
            )

    return run


# --- ported from pyspark/worker.py:325-367 ---
def verify_iterator_exhausted(iterator: Iterator, error_class: str) -> None:
    """Verify that an iterator has been fully consumed."""
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise PySparkRuntimeError(errorClass=error_class, messageParameters={})


def verify_output_row_limit(iterator: Iterator, max_rows: Callable[[], int], error_class: str) -> Iterator:
    """Yield elements while verifying total rows do not exceed a limit (fail-fast)."""
    total_rows = 0
    for element in iterator:
        total_rows += len(element)
        if total_rows > max_rows():
            raise PySparkRuntimeError(errorClass=error_class, messageParameters={})
        yield element


def verify_output_row_count(iterator: Iterator, expected_rows: Callable[[], int], error_class: str) -> Iterator:
    """Yield elements and verify final row count matches expected exactly."""
    actual_rows = 0
    for element in iterator:
        actual_rows += len(element)
        yield element

    expected = expected_rows()
    if actual_rows != expected:
        raise PySparkRuntimeError(
            errorClass=error_class,
            messageParameters={
                "output_length": str(actual_rows),
                "input_length": str(expected),
            },
        )


# --- SQL_SCALAR_ARROW_ITER_UDF (251): worker.py:2586-2646 ---
def _scalar_arrow_iter(f: Callable[..., Any], return_type: Any) -> Runner:
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowBatchTransformer
    from krishiv._udf.pandas_types import to_arrow_type

    udf_func = fail_on_stopiteration(f)

    def run(source: FrameSource, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        schema = _column_schema(return_type, conf)
        arrow_return_type = to_arrow_type(return_type, timezone="UTC", prefers_large_types=conf.use_large_var_types)
        num_input_rows = 0

        def extract_args(batch: Any) -> Any:
            nonlocal num_input_rows
            args = tuple(batch.column(o) for o in range(batch.num_columns))
            num_input_rows += batch.num_rows
            return args[0] if len(args) == 1 else args

        args_iter = map(extract_args, _arg_batches(source, conf))
        verified_iter = verify_result(pa.Array)(udf_func(args_iter))
        target_schema = pa.schema([pa.field("_0", arrow_return_type)])

        def process_results() -> Iterator[Any]:
            for result in verified_iter:
                batch = pa.RecordBatch.from_arrays([result], ["_0"])
                yield ArrowBatchTransformer.enforce_schema(batch, target_schema, safecheck=True)

        limited = verify_output_row_limit(
            process_results(), lambda: num_input_rows, error_class="OUTPUT_EXCEEDS_INPUT_ROWS"
        )
        matched = verify_output_row_count(limited, lambda: num_input_rows, error_class="RESULT_ROWS_MISMATCH")
        for batch in matched:
            yield _from_arrow(batch, schema)
        verify_iterator_exhausted(args_iter, error_class="INPUT_NOT_FULLY_CONSUMED")

    return run


def make_relation(f: Callable[..., Any], eval_type: int, return_type: Any) -> Runner:
    """The runner of ``f`` for the relation kind ``eval_type`` (``read_single_udf``, ``worker.py:1064-1188``,
    then ``read_udfs``'s loop for that kind)."""
    if eval_type == SQL_MAP_PANDAS_ITER_UDF:
        return _map_pandas(f, return_type)
    if eval_type == SQL_MAP_ARROW_ITER_UDF:
        return _map_arrow(f, return_type)
    if eval_type == SQL_GROUPED_MAP_PANDAS_UDF:
        return _grouped_map_pandas(f, return_type)
    if eval_type == SQL_GROUPED_MAP_PANDAS_ITER_UDF:
        return _grouped_map_pandas_iter(f, return_type)
    if eval_type in (SQL_GROUPED_MAP_ARROW_UDF, SQL_GROUPED_MAP_ARROW_ITER_UDF) and isinstance(return_type, StructType):
        return _grouped_map_arrow(f, return_type, eval_type == SQL_GROUPED_MAP_ARROW_ITER_UDF)
    if eval_type == SQL_COGROUPED_MAP_PANDAS_UDF:
        return _cogrouped_map_pandas(f, return_type)
    if eval_type == SQL_COGROUPED_MAP_ARROW_UDF and isinstance(return_type, StructType):
        return _cogrouped_map_arrow(f, return_type)
    if eval_type in (SQL_GROUPED_AGG_PANDAS_UDF, SQL_GROUPED_AGG_PANDAS_ITER_UDF):
        return _grouped_agg_pandas(f, return_type, eval_type == SQL_GROUPED_AGG_PANDAS_ITER_UDF)
    if eval_type in (SQL_GROUPED_AGG_ARROW_UDF, SQL_GROUPED_AGG_ARROW_ITER_UDF):
        return _grouped_agg_arrow(f, return_type, eval_type == SQL_GROUPED_AGG_ARROW_ITER_UDF)
    if eval_type == SQL_WINDOW_AGG_PANDAS_UDF:
        return _window_agg_pandas(f, return_type)
    if eval_type == SQL_WINDOW_AGG_ARROW_UDF:
        return _window_agg_arrow(f, return_type)
    if eval_type == SQL_SCALAR_PANDAS_ITER_UDF:
        return _scalar_pandas_iter(f, return_type)
    if eval_type == SQL_SCALAR_ARROW_ITER_UDF:
        return _scalar_arrow_iter(f, return_type)
    from krishiv._udf._host import not_implemented

    raise not_implemented(f"Python UDFs of eval type {eval_type} as a relation")
