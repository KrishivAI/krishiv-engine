# SPDX-License-Identifier: Apache-2.0
# Parts ported from Apache Spark v4.2.0, python/pyspark/worker.py (read_udtf, L1191-2351) and
# python/pyspark/sql/worker/analyze_udtf.py (Apache License 2.0; see NOTICE). Each ported function names its
# source lines; the code keeps Spark's shape so a later diff against the tag stays readable.
"""Python table functions (UDTFs: ``@udtf`` 300, ``@udtf(useArrow=True)`` 301, ``@arrow_udtf`` 302) over the
relation frames (design ``P3-T085-embedded-python-udf-kinds.md`` §2, §3.2), and their ``analyze``.

A UDTF task is one Spark task: its frames are batches of the call's argument columns (Spark's ``allInputs``:
each distinct argument once, a ``TABLE`` argument as one struct column holding the table row, the projected
``PARTITION BY`` values among its fields), then ``end``. :func:`make_udtf_runner` calls ``eval`` per row (per
batch for 302), ``terminate`` at each ``PARTITION BY`` boundary (found here, as Spark's wrapper finds it, by
comparing the partition fields of consecutive rows) and at the end, and ``cleanup`` last, and yields the result
rows as batches of the declared struct's fields in the engine's Arrow forms.

The call comes in the task message (:class:`UdtfCall`): the argument offsets (by position, or by name for a
named argument), the partition child indexes, the declared return type (Spark JSON) and ``analyze``'s pickled
``AnalyzeResult`` when the UDTF has one (passed to ``__init__`` as Spark does).

A 300 UDTF's rows are converted as Spark's are: ``StructType.toInternal`` in the worker (``wrap_udtf``), then, for
what the JVM does with the pickled row, :func:`_from_java`: Pyrolite's view of each Python value (a 32-bit
``int`` is an ``Integer``, a larger one a ``Long``, ...) through ``EvaluatePython.makeFromJava`` (v4.2.0): a value
of another type than the declared one is mostly NULL, an integer is truncated to the column's width, anything
is a string's ``toString``; a short tuple for a struct is ``STRUCT_ARRAY_LENGTH_MISMATCH`` and a decimal over its
precision ``NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION`` (:class:`SparkJvmError`, which the worker reports as
those classes). A JVM object's own ``toString`` (``[B@...`` for bytes, a ``GregorianCalendar`` for a date, an
``Object[]`` for a tuple) is not reproduced: the value's Python ``str`` stands for it.

Not here: Spark's legacy pandas conversion of 301
(``spark.sql.legacy.execution.pythonUDTF.pandas.conversion.enabled``), refused with ``NOT_IMPLEMENTED``.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from krishiv._udf._host import IN_WORKER, PySparkRuntimeError, PySparkTypeError, PySparkValueError, not_implemented
from krishiv._udf.types import Row, StructType, _create_row
from krishiv._udf.worker import RunnerConf, _engine_arrow_type, _runner_conf, _spark_arrow_type, _to_spark

#: Spark's ``PythonEvalType`` codes of the table kinds (``pyspark/util.py:655-688``).
SQL_TABLE_UDF = 300
SQL_ARROW_TABLE_UDF = 301
SQL_ARROW_UDTF = 302

#: The UDTF kinds :func:`make_udtf_runner` runs.
UDTF_EVAL_TYPES = frozenset({SQL_TABLE_UDF, SQL_ARROW_TABLE_UDF, SQL_ARROW_UDTF})

#: The schema metadata key of an ``analyze`` call's input batch: the call's name and arguments, as JSON
#: (``krishiv-udf-pool`` ``udtf.rs`` ``analyze_input``).
ANALYZE_METADATA = b"krishiv.udtf.analyze"

#: Spark's legacy pandas conversion of an Arrow UDTF (``SQLConf.PYTHON_UDTF_LEGACY_PANDAS_CONVERSION``).
LEGACY_PANDAS_CONVERSION = "spark.sql.legacy.execution.pythonUDTF.pandas.conversion.enabled"


def _skip_rest_exception() -> type[BaseException]:
    """``pyspark.sql.udtf.SkipRestOfInputTableException``: the class a UDTF raises (pyspark's in a pool worker)."""
    if IN_WORKER:
        from pyspark.sql import udtf  # ty: ignore[unresolved-import]  # pool workers only

        # PySpark 3.5 has no such class, so a 3.5 UDTF cannot raise it.
        return getattr(udtf, "SkipRestOfInputTableException", _SkipRestOfInputTable)
    return _SkipRestOfInputTable


class _SkipRestOfInputTable(Exception):
    """Stands in for ``SkipRestOfInputTableException`` in-process (the facade has no ``udtf`` module yet)."""


def _parse_type(text: str) -> Any:
    if IN_WORKER:
        from pyspark.sql.types import _parse_datatype_json_string  # ty: ignore[unresolved-import]
    else:
        from krishiv.sql.types import _parse_datatype_json_string
    return _parse_datatype_json_string(text)


@dataclasses.dataclass(frozen=True)
class UdtfCall:
    """One UDTF call site, as the task message carries it (``udtf``)."""

    #: The UDTF's name, for errors.
    name: str
    #: The declared (or analyzed) return type: a struct, in Spark's JSON.
    return_type: str
    #: ``(offset, name)`` per argument of the call: the frame column it reads, and its name when named.
    args: list[tuple[int, str | None]]
    #: The fields of the ``TABLE`` argument's struct that are projected ``PARTITION BY`` values.
    partition_child_indexes: list[int]
    #: The frame columns that are ``TABLE`` arguments (302 flattens them into record batches).
    table_arg_offsets: list[int]
    #: ``analyze``'s pickled ``AnalyzeResult``, hex, when the UDTF has an ``analyze`` method.
    analyze_result: str | None
    #: A ``LATERAL`` call's input columns: the trailing frame columns, read by no argument; each result row
    #: leads with the values of the input row whose ``eval`` gave it (``EvalPythonUDTFExec``'s join).
    passthrough: list[int] = dataclasses.field(default_factory=list)

    @staticmethod
    def of(msg: dict[str, Any]) -> UdtfCall:
        return UdtfCall(
            name=str(msg["name"]),
            return_type=str(msg["return_type"]),
            args=[(int(o), None if n is None else str(n)) for o, n in msg.get("args") or []],
            partition_child_indexes=[int(i) for i in msg.get("partition_child_indexes") or []],
            table_arg_offsets=[int(i) for i in msg.get("table_arg_offsets") or []],
            analyze_result=msg.get("analyze_result"),
            passthrough=[int(i) for i in msg.get("passthrough") or []],
        )


class _Cursor:
    """The input row a ``LATERAL`` call's results join with: the current input batch, and the batch and row of
    the last ``eval`` call (``None`` before any)."""

    def __init__(self) -> None:
        self.raw: Any = None
        self.ref: tuple[Any, int] | None = None


class SparkJvmError(Exception):
    """An error Spark's JVM raises converting a 300 UDTF's row (``EvaluatePython.makeFromJava``): the worker
    reports its class and parameters, and the server raises that class."""

    def __init__(self, condition: str, parameters: dict[str, str]) -> None:
        super().__init__(f"[{condition}] {parameters}")
        self.condition = condition
        self.parameters = parameters


def _wrap_kwargs_support(f: Callable[..., Any], args_offsets: list[int], kwargs_offsets: dict[str, int]) -> Any:
    """``wrap_kwargs_support`` (``worker.py:956-973``)."""
    if len(kwargs_offsets):
        keys = list(kwargs_offsets.keys())

        len_args_offsets = len(args_offsets)
        if len_args_offsets > 0:

            def func(*args: Any) -> Any:
                return f(*args[:len_args_offsets], **dict(zip(keys, args[len_args_offsets:], strict=True)))

        else:

            def func(*args: Any) -> Any:
                return f(**dict(zip(keys, args, strict=True)))

        return func, args_offsets + [kwargs_offsets[key] for key in keys]
    return f, args_offsets


def _handler_factory(handler: Any, call: UdtfCall) -> Callable[[], Any]:
    """The UDTF class's constructor, with ``analyze``'s result when it has one (``worker.py:1233-1270``)."""
    import pickle

    if not isinstance(handler, type):
        raise PySparkRuntimeError(
            f"Invalid UDTF handler type. Expected a class (type 'type'), but "
            f"got an instance of {type(handler).__name__}."
        )
    udtf_init_args = inspect.getfullargspec(handler)
    if call.analyze_result is not None:
        pickled_analyze_result = pickle.loads(bytes.fromhex(call.analyze_result))
        if len(udtf_init_args.args) > 2:
            raise PySparkRuntimeError(
                errorClass="UDTF_CONSTRUCTOR_INVALID_IMPLEMENTS_ANALYZE_METHOD",
                messageParameters={"name": call.name},
            )
        if len(udtf_init_args.args) == 2:
            prev_handler = handler

            def construct_udtf() -> Any:
                # Here we pass the AnalyzeResult to the UDTF's __init__ method.
                return prev_handler(dataclasses.replace(pickled_analyze_result))

            return construct_udtf
    elif len(udtf_init_args.args) > 1:
        raise PySparkRuntimeError(
            errorClass="UDTF_CONSTRUCTOR_INVALID_NO_ANALYZE_METHOD",
            messageParameters={"name": call.name},
        )
    return handler


class _UDTFWithPartitions:
    """``UDTFWithPartitions`` (``worker.py:1272-1377``): a UDTF with a ``TABLE`` argument and ``PARTITION BY``:
    ``terminate`` and a fresh instance where the partition values of consecutive rows change."""

    def __init__(self, create_udtf: Callable[[], Any], partition_child_indexes: list[int]) -> None:
        self._create_udtf = create_udtf
        self._udtf = create_udtf()
        self._prev_arguments: list[Any] = []
        self._partition_child_indexes = partition_child_indexes
        self._eval_raised_skip_rest_of_input_table = False
        self._skip = _skip_rest_exception()

    def eval(self, *args: Any, **kwargs: Any) -> Iterator:
        changed_partitions = self._check_partition_boundaries(list(args) + list(kwargs.values()))
        if changed_partitions:
            if hasattr(self._udtf, "terminate"):
                result = self._udtf.terminate()
                if result is not None:
                    yield from result
            self._udtf = self._create_udtf()
            self._eval_raised_skip_rest_of_input_table = False
        if self._udtf.eval is not None and not self._eval_raised_skip_rest_of_input_table:
            # Filter the arguments to exclude projected PARTITION BY values added by Catalyst.
            filtered_args = [self._remove_partition_by_exprs(arg) for arg in args]
            filtered_kwargs = {key: self._remove_partition_by_exprs(value) for (key, value) in kwargs.items()}
            try:
                result = self._udtf.eval(*filtered_args, **filtered_kwargs)
                if result is not None:
                    yield from result
            except self._skip:
                self._eval_raised_skip_rest_of_input_table = True

    def terminate(self) -> Iterator:
        if hasattr(self._udtf, "terminate"):
            return self._udtf.terminate()
        return iter(())

    def cleanup(self) -> None:
        if hasattr(self._udtf, "cleanup"):
            self._udtf.cleanup()

    def _check_partition_boundaries(self, arguments: list[Any]) -> bool:
        result = False
        if len(self._prev_arguments) > 0:
            cur_table_arg = self._get_table_arg(arguments)
            prev_table_arg = self._get_table_arg(self._prev_arguments)
            cur = [cur_table_arg[i] for i in self._partition_child_indexes]
            prev = [prev_table_arg[i] for i in self._partition_child_indexes]
            result = any(k != v for k, v in zip(cur, prev, strict=True))
        self._prev_arguments = arguments
        return result

    def _get_table_arg(self, inputs: list[Any]) -> Any:
        return next(x for x in inputs if type(x) is Row)

    def _remove_partition_by_exprs(self, arg: Any) -> Any:
        if isinstance(arg, Row):
            new_row_keys = []
            new_row_values = []
            for i, (key, value) in enumerate(zip(arg.__fields__, arg, strict=True)):
                if i not in self._partition_child_indexes:
                    new_row_keys.append(key)
                    new_row_values.append(value)
            return _create_row(new_row_keys, new_row_values)
        return arg


class _ArrowUDTFWithPartition:
    """``ArrowUDTFWithPartition`` (``worker.py:1379-1690``): an Arrow UDTF (302) with ``PARTITION BY``: each
    batch is cut where the partition values change, ``eval`` called per slice, ``terminate`` and a fresh
    instance at a change."""

    def __init__(self, create_udtf: Callable[[], Any], partition_child_indexes: list[int]) -> None:
        self._create_udtf = create_udtf
        self._udtf = create_udtf()
        self._partition_child_indexes = partition_child_indexes
        self._last_partition_key: tuple[Any, ...] | None = None
        self._eval_raised_skip_rest_of_input_table = False
        self._skip = _skip_rest_exception()

    def eval(self, *args: Any, **kwargs: Any) -> Iterator:
        import pyarrow as pa

        original_batch = self._get_table_arg(list(args) + list(kwargs.values()))
        if not isinstance(original_batch, pa.RecordBatch):
            raise PySparkRuntimeError(
                errorClass="INVALID_ARROW_UDTF_TABLE_ARGUMENT",
                messageParameters={
                    "actual_type": (str(type(original_batch)) if original_batch is not None else "None")
                },
            )
        filtered_args = [self._remove_partition_by_exprs(arg) for arg in args]
        filtered_kwargs = {key: self._remove_partition_by_exprs(value) for (key, value) in kwargs.items()}
        filtered_batch = self._get_table_arg(filtered_args + list(filtered_kwargs.values()))
        yield from self._process_arrow_batch_by_partitions(
            original_batch, filtered_batch, filtered_args, filtered_kwargs
        )

    def _process_arrow_batch_by_partitions(
        self, original_batch: Any, filtered_batch: Any, filtered_args: list[Any], filtered_kwargs: dict[str, Any]
    ) -> Iterator:
        import pyarrow as pa

        boundaries = self._detect_partition_boundaries(original_batch)
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            partition_key = tuple(
                original_batch.column(idx)[start_idx].as_py() for idx in self._partition_child_indexes
            )
            is_new_partition = self._last_partition_key is not None and partition_key != self._last_partition_key
            if is_new_partition:
                if hasattr(self._udtf, "terminate"):
                    terminate_result = self._udtf.terminate()
                    if terminate_result is not None:
                        yield from terminate_result
                self._udtf = self._create_udtf()
                self._eval_raised_skip_rest_of_input_table = False
            partition_batch = filtered_batch.slice(start_idx, end_idx - start_idx)
            self._last_partition_key = partition_key
            partition_filtered_args = [
                partition_batch if isinstance(arg, pa.RecordBatch) else arg for arg in filtered_args
            ]
            partition_filtered_kwargs = {
                key: partition_batch if isinstance(value, pa.RecordBatch) else value
                for key, value in filtered_kwargs.items()
            }
            if not self._eval_raised_skip_rest_of_input_table:
                try:
                    result = self._udtf.eval(*partition_filtered_args, **partition_filtered_kwargs)
                    if result is not None:
                        yield from result
                except self._skip:
                    self._eval_raised_skip_rest_of_input_table = True

    def terminate(self) -> Iterator:
        if hasattr(self._udtf, "terminate"):
            return self._udtf.terminate()
        return iter(())

    def cleanup(self) -> None:
        if hasattr(self._udtf, "cleanup"):
            self._udtf.cleanup()

    def _get_table_arg(self, inputs: list[Any]) -> Any:
        import pyarrow as pa

        batches = [arg for arg in inputs if isinstance(arg, pa.RecordBatch)]
        if len(batches) == 0:
            return None
        if len(batches) == 1:
            return batches[0]
        raise RuntimeError(
            f"Expected exactly one pa.RecordBatch argument for TABLE parameter, "
            f"but found {len(batches)}. Received types: "
            f"{[type(arg).__name__ for arg in inputs]}"
        )

    def _detect_partition_boundaries(self, batch: Any) -> list[int]:
        boundaries = [0]
        if batch.num_rows <= 1:
            boundaries.append(batch.num_rows)
            return boundaries
        partition_arrays = [batch.column(i) for i in self._partition_child_indexes]
        for row_idx in range(1, batch.num_rows):
            if any(col[row_idx].as_py() != col[row_idx - 1].as_py() for col in partition_arrays):
                boundaries.append(row_idx)
        boundaries.append(batch.num_rows)
        return boundaries

    def _remove_partition_by_exprs(self, arg: Any) -> Any:
        import pyarrow as pa

        if isinstance(arg, pa.RecordBatch):
            keep_indices = [i for i in range(len(arg.schema.names)) if i not in self._partition_child_indexes]
            if keep_indices:
                return pa.RecordBatch.from_arrays(
                    [arg.column(i) for i in keep_indices], names=[arg.schema.names[i] for i in keep_indices]
                )
            # Spark builds this with `from_arrays([], ..., num_rows=...)`, which pyarrow 26 does not take.
            return arg.select([])
        return arg


def _build_null_checker(return_type: StructType) -> Callable[[Any], None] | None:
    """``build_null_checker`` (``worker.py:1733-1837``): a check of each result row against the non-nullable
    fields of ``return_type``, or ``None`` when every field is nullable all the way down."""
    from krishiv._udf.types import ArrayType, MapType

    def raise_(result_column_index: int) -> None:
        raise PySparkRuntimeError(
            errorClass="UDTF_EXEC_ERROR",
            messageParameters={
                "method_name": "eval' or 'terminate",
                "error": f"Column {result_column_index} within a returned row had a "
                + "value of None, either directly or within array/struct/map "
                + "subfields, but the corresponding column type was declared as "
                + "non-nullable; please update the UDTF to return a non-None value at "
                + "this location or otherwise declare the column type as nullable.",
            },
        )

    def checker(data_type: Any, result_column_index: int) -> Callable[[Any], None] | None:
        if isinstance(data_type, ArrayType):
            element_checker = checker(data_type.elementType, result_column_index)
            contains_null = data_type.containsNull
            if element_checker is None and contains_null:
                return None

            def check_array(arr: Any) -> None:
                if isinstance(arr, list):
                    for e in arr:
                        if e is None:
                            if not contains_null:
                                raise_(result_column_index)
                        elif element_checker is not None:
                            element_checker(e)

            return check_array
        if isinstance(data_type, MapType):
            key_checker = checker(data_type.keyType, result_column_index)
            value_checker = checker(data_type.valueType, result_column_index)
            value_contains_null = data_type.valueContainsNull

            def check_map(map: Any) -> None:
                if isinstance(map, dict):
                    for k, v in map.items():
                        if k is None:
                            raise_(result_column_index)
                        elif key_checker is not None:
                            key_checker(k)
                        if value_checker is None and value_contains_null:
                            continue
                        if v is None:
                            if not value_contains_null:
                                raise_(result_column_index)
                        elif value_checker is not None:
                            value_checker(v)

            return check_map
        if isinstance(data_type, StructType):
            field_checkers = [checker(f.dataType, result_column_index) for f in data_type]
            nullables = [f.nullable for f in data_type]
            if all(c is None for c in field_checkers) and all(nullables):
                return None

            def check_struct(struct: Any) -> None:
                if isinstance(struct, tuple):
                    for value, check, nullable in zip(struct, field_checkers, nullables, strict=False):
                        if value is None:
                            if not nullable:
                                raise_(result_column_index)
                        elif check is not None:
                            check(value)

            return check_struct
        return None

    field_checkers = [checker(f.dataType, result_column_index=i) for i, f in enumerate(return_type)]
    nullables = [f.nullable for f in return_type]
    if all(c is None for c in field_checkers) and all(nullables):
        return None

    def check(row: Any) -> None:
        if isinstance(row, tuple):
            for i, (value, field_check, nullable) in enumerate(zip(row, field_checkers, nullables, strict=False)):
                if value is None:
                    if not nullable:
                        raise_(i)
                elif field_check is not None:
                    field_check(value)

    return check


def _engine_schema(return_type: StructType, conf: RunnerConf) -> Any:
    """The engine's Arrow schema of the result: the struct's fields in the engine's forms."""
    import pyarrow as pa

    from krishiv._udf.pandas_types import to_arrow_type

    return pa.schema(
        [
            pa.field(
                f.name,
                _engine_arrow_type(
                    _spark_arrow_type(
                        to_arrow_type(f.dataType, timezone="UTC", prefers_large_types=conf.use_large_var_types), "UTC"
                    )
                ),
            )
            for f in return_type.fields
        ]
    )


def _to_engine(batch: Any, schema: Any) -> Any:
    """``batch`` (Spark's Arrow forms of the declared fields) in the engine's forms."""
    import pyarrow as pa

    pairs = zip(batch.columns, schema, strict=True)
    arrays = [col if col.type == field.type else col.cast(field.type) for col, field in pairs]
    return pa.RecordBatch.from_arrays(arrays, schema=schema) if arrays else batch


def make_udtf_runner(handler: Any, eval_type: int, call: UdtfCall) -> Callable[[Any, int, int], Iterator[Any]]:
    """The runner of UDTF ``handler`` for ``eval_type`` at ``call`` (``read_udtf``, ``worker.py:1191-2351``)."""
    from krishiv._udf.relations import _map_batches

    return_type = _parse_type(call.return_type)
    if not isinstance(return_type, StructType):
        raise PySparkRuntimeError(f"The return type of a UDTF must be a struct type, but got {type(return_type)}.")
    create = _handler_factory(handler, call)
    args_offsets = [o for o, n in call.args if n is None]
    kwargs_offsets = {n: o for o, n in call.args if n is not None}

    def instantiate() -> Any:
        # Instantiate the UDTF class (worker.py:1691-1704).
        try:
            if len(call.partition_child_indexes) > 0:
                if eval_type == SQL_ARROW_UDTF:
                    return _ArrowUDTFWithPartition(create, call.partition_child_indexes)
                return _UDTFWithPartitions(create, call.partition_child_indexes)
            return create()
        except Exception as e:
            raise PySparkRuntimeError(
                errorClass="UDTF_EXEC_ERROR",
                messageParameters={"method_name": "__init__", "error": str(e)},
            ) from None

    def run(source: Any, key_count: int, arity: int) -> Iterator[Any]:
        conf = _runner_conf()
        if eval_type == SQL_ARROW_TABLE_UDF and conf.get(LEGACY_PANDAS_CONVERSION) == "true":
            raise not_implemented(f"{LEGACY_PANDAS_CONVERSION}=true for an Arrow Python UDTF")
        udtf = instantiate()
        # Validate the UDTF (worker.py:1706-1722).
        if not hasattr(udtf, "eval"):
            raise PySparkRuntimeError(
                "Failed to execute the user defined table function because it has not "
                "implemented the 'eval' method. Please add the 'eval' method and try "
                "the query again."
            )
        try:
            inspect.signature(udtf.eval).bind(*args_offsets, **kwargs_offsets)
        except TypeError as e:
            raise PySparkRuntimeError(
                errorClass="UDTF_EVAL_METHOD_ARGUMENTS_DO_NOT_MATCH_SIGNATURE",
                messageParameters={"name": call.name, "reason": str(e)},
            ) from None
        schema = _engine_schema(return_type, conf)
        carried = len(call.passthrough)
        cursor = _Cursor()

        def spark_batches() -> Iterator[Any]:
            for raw in _map_batches(source):
                cursor.raw = raw
                args = raw if carried == 0 else raw.select(list(range(raw.num_columns - carried)))
                yield _spark_batch(args, conf)

        if carried:
            if eval_type == SQL_ARROW_UDTF:
                raise PySparkRuntimeError("an Arrow UDTF (302) cannot be a LATERAL call")
            for batch in _row_udtf_mapper(
                udtf, eval_type, return_type, call, args_offsets, kwargs_offsets, conf, spark_batches(), cursor
            ):
                yield _carry(_to_engine(batch, schema), cursor, carried)
            return
        mapper = _arrow_udtf_mapper if eval_type == SQL_ARROW_UDTF else _row_udtf_mapper
        for batch in mapper(udtf, eval_type, return_type, call, args_offsets, kwargs_offsets, conf, spark_batches()):
            yield _to_engine(batch, schema)

    return run


def _carry(batch: Any, cursor: _Cursor, carried: int) -> Any:
    """``batch`` led by the ``carried`` trailing input columns of the row ``cursor`` names (all null before any
    row: Krishiv's answer where Spark 4.2.0 fails with a ``NullPointerException``)."""
    import pyarrow as pa

    raw = cursor.ref[0] if cursor.ref is not None else cursor.raw
    if raw is None:
        # No input batch came (the engine opens no task for an empty input): nothing to lead with.
        raise PySparkRuntimeError("a LATERAL Python UDTF produced rows before any input batch")
    fields = list(raw.schema)[raw.num_columns - carried :]
    n = batch.num_rows
    if cursor.ref is None:
        cols = [pa.nulls(n, type=f.type) for f in fields]
    else:
        row = cursor.ref[1]
        # The input row repeated by slicing, not ``take``: pyarrow has no ``take`` kernel for the view types
        # (``string_view``, ``binary_view``) the engine sends.
        cols = [_repeat_row(raw.column(raw.num_columns - carried + j), row, n) for j in range(carried)]
    out = pa.schema([pa.field(f.name, f.type) for f in fields] + list(batch.schema))
    return pa.RecordBatch.from_arrays(cols + list(batch.columns), schema=out)


def _repeat_row(column: Any, row: int, n: int) -> Any:
    """``column``'s value at ``row``, ``n`` times, of ``column``'s type (any Arrow type)."""
    import pyarrow as pa

    one = column.slice(row, 1)
    if n == 0:
        return column.slice(row, 0)
    return pa.concat_arrays([one] * n)


def _spark_batch(batch: Any, conf: RunnerConf) -> Any:
    import pyarrow as pa

    if batch.num_columns == 0:
        return batch  # a call without arguments: only the row count matters

    return pa.RecordBatch.from_arrays(
        [_to_spark(c, conf.timezone) for c in batch.columns], names=list(batch.schema.names)
    )


def _row_udtf_mapper(
    udtf: Any,
    eval_type: int,
    return_type: StructType,
    call: UdtfCall,
    args_offsets: list[int],
    kwargs_offsets: dict[str, int],
    conf: RunnerConf,
    batches: Iterable[Any],
    cursor: _Cursor | None = None,
) -> Iterator[Any]:
    """300 and 301 (the non-legacy path, ``worker.py:1946-2135``; 300's ``wrap_udtf`` checks,
    ``worker.py:2228-2349``): ``eval`` per row, the result rows converted to Arrow."""
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowTableToRowsConversion, LocalDataToArrowConversion
    from krishiv._udf.pandas_types import from_arrow_type, to_arrow_type

    skip = _skip_rest_exception()
    check_output_row_against_schema = _build_null_checker(return_type)
    arrow_return_type = to_arrow_type(return_type, timezone="UTC", prefers_large_types=conf.use_large_var_types)
    return_type_size = len(return_type)
    # Only 300 reads it, in a pool worker, where the type is pyspark's (the facade runs no UDTF).
    to_internal: Callable[[Any], Any] = getattr(return_type, "toInternal", lambda row: row)

    def wrap(f: Callable[..., Any]) -> Callable[..., Iterator[Any]]:
        fname = getattr(f, "__name__", "")

        def func(*a: Any) -> Any:
            try:
                return f(*a)
            except skip:
                raise
            except Exception as e:
                raise PySparkRuntimeError(
                    errorClass="UDTF_EXEC_ERROR",
                    messageParameters={"method_name": fname, "error": str(e)},
                ) from None

        def check_return_value(res: Any) -> Iterator[Any]:
            if res is not None:
                if not isinstance(res, Iterable):
                    raise PySparkRuntimeError(
                        errorClass="UDTF_RETURN_NOT_ITERABLE",
                        messageParameters={"type": type(res).__name__, "func": fname},
                    )
                for row in res:
                    if eval_type == SQL_TABLE_UDF:
                        _check_table_udf_row(row, return_type_size, fname)
                    elif not isinstance(row, tuple) and return_type_size == 1:
                        row = (row,)
                    if check_output_row_against_schema is not None and row is not None:
                        check_output_row_against_schema(row)
                    yield row

        def convert_to_arrow(data: Any) -> Any:
            rows = list(check_return_value(data))
            if eval_type == SQL_TABLE_UDF:
                # `wrap_udtf`'s `toInternal`, then the JVM's conversion; a None row stays None (the JVM joins it
                # with a null row, EvalPythonUDTFExec).
                internal = [None if r is None else to_internal(r) for r in rows]
                return _jvm_batch(internal, return_type, arrow_return_type)
            if len(rows) == 0:
                return pa.RecordBatch.from_pylist([], schema=pa.schema(list(arrow_return_type)))
            try:
                table = LocalDataToArrowConversion.convert(rows, return_type, conf.use_large_var_types)
            except PySparkValueError as e:
                if e.getErrorClass() == "AXIS_LENGTH_MISMATCH":
                    params = e.getMessageParameters() or {}
                    raise PySparkRuntimeError(
                        errorClass="UDTF_RETURN_SCHEMA_MISMATCH",
                        messageParameters={
                            "expected": params["expected_length"],
                            "actual": params["actual_length"],
                            "func": fname,
                        },
                    ) from e
                raise _conversion_error(rows, return_type, arrow_return_type) from e
            except Exception as e:
                raise _conversion_error(rows, return_type, arrow_return_type) from e
            return pa.RecordBatch.from_arrays(
                [c.combine_chunks() for c in table.columns], schema=pa.schema(list(arrow_return_type))
            )

        def evaluate(*args: Any) -> Iterator[Any]:
            yield convert_to_arrow(func(*args))

        return evaluate

    eval_func, args_kwargs_offsets = _wrap_kwargs_support(udtf.eval, args_offsets, kwargs_offsets)
    evaluate = wrap(eval_func)
    terminate = wrap(udtf.terminate) if hasattr(udtf, "terminate") else None
    cleanup = udtf.cleanup if hasattr(udtf, "cleanup") else None
    try:
        for batch in batches:
            converters = [
                ArrowTableToRowsConversion._create_converter(
                    from_arrow_type(col.type), none_on_identity=True, binary_as_bytes=conf.binary_as_bytes
                )
                for col in batch.columns
            ]
            pylist = [
                (
                    [conv(v) for v in ArrowTableToRowsConversion._to_pylist(column)]
                    if conv is not None
                    else ArrowTableToRowsConversion._to_pylist(column)
                )
                for column, conv in zip(batch.columns, converters, strict=True)
            ]
            if args_kwargs_offsets:
                for i, row in enumerate(zip(*[pylist[o] for o in args_kwargs_offsets], strict=True)):
                    if cursor is not None:
                        cursor.ref = (cursor.raw, i)
                    yield from evaluate(*row)
            else:
                for i in range(batch.num_rows):
                    if cursor is not None:
                        cursor.ref = (cursor.raw, i)
                    yield from evaluate()
        if terminate is not None:
            yield from terminate()
    except skip:
        if terminate is not None:
            yield from terminate()
    finally:
        if cleanup is not None:
            cleanup()


def _java_kind(v: Any) -> str:
    """The Java type Pyrolite unpickles ``v`` as (``net.razorvine.pickle``), by what ``makeFromJava`` matches."""
    import datetime
    import decimal

    if isinstance(v, bool):
        return "Boolean"
    if isinstance(v, int):
        if -(2**31) <= v < 2**31:
            return "Integer"
        return "Long" if -(2**63) <= v < 2**63 else "BigInteger"
    if isinstance(v, float):
        return "Double"
    if isinstance(v, str):
        return "String"
    if isinstance(v, (bytes, bytearray)):
        return "byte[]"
    if isinstance(v, decimal.Decimal):
        return "BigDecimal"
    if isinstance(v, (datetime.date, datetime.time)):
        return "Calendar"
    if isinstance(v, tuple):
        return "Object[]"
    if isinstance(v, list):
        return "List"
    if isinstance(v, dict):
        return "Map"
    return "Object"


def _wrap(v: int, bits: int) -> int:
    """Java's narrowing of an integer to ``bits`` (``Long.toInt``: the low bits, two's complement)."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >= 1 << (bits - 1) else v


def _java_string(v: Any) -> str:
    """Java's ``toString`` of the unpickled value, where Python's text is the same; else Python's ``str``."""
    kind = _java_kind(v)
    if kind == "Boolean":
        return "true" if v else "false"
    if kind == "List":
        return "[" + ", ".join("null" if e is None else _java_string(e) for e in v) + "]"
    if kind == "Map":
        return "{" + ", ".join(f"{_java_text(k)}={_java_text(x)}" for k, x in _hash_map_order(v)) + "}"
    if kind == "Double":
        return _java_double(v)
    return str(v)


def _java_hash(v: Any) -> int | None:
    """The Java ``hashCode`` (as an unsigned 32-bit int) of the key Pyrolite unpickles ``v`` as; ``None`` for a
    key whose hash is identity-based or not modelled here (``byte[]``, ``Object[]``, ``BigInteger``, ...)."""
    import struct

    mask = 0xFFFFFFFF
    if v is None:
        return 0
    kind = _java_kind(v)
    if kind == "Boolean":
        return 1231 if v else 1237
    if kind == "Integer":
        return v & mask
    if kind == "Long":
        u = v & 0xFFFFFFFFFFFFFFFF
        return (u ^ (u >> 32)) & mask
    if kind == "Double":
        # `Double.hashCode`: `doubleToLongBits` (one canonical NaN), its halves XORed.
        bits = 0x7FF8000000000000 if v != v else struct.unpack(">Q", struct.pack(">d", v))[0]
        return (bits ^ (bits >> 32)) & mask
    if kind == "String":
        h = 0
        try:
            data = v.encode("utf-16-be")
        except UnicodeEncodeError:
            # A lone surrogate: Python pickles it with `surrogatepass` and Pyrolite decodes those bytes as
            # malformed UTF-8, so the Java key is not this text and its hash is not modelled.
            return None
        for i in range(0, len(data), 2):
            h = (31 * h + ((data[i] << 8) | data[i + 1])) & mask
        return h
    return None


def _hash_map_order(m: dict[Any, Any]) -> list[tuple[Any, Any]]:
    """``m``'s entries in the order ``java.util.HashMap`` iterates them once Pyrolite has put them, in Python's
    order, into a default ``HashMap`` (``Unpickler.load_empty_dictionary``): by bucket (the spread hash
    ``h ^ (h >>> 16)`` over the table), then by insertion within a bucket. The puts are replayed: the table
    starts at 16 buckets and doubles when a put makes it more than 3/4 full, and also when a put makes one bin
    9 nodes long while the table has fewer than 64 buckets (``putVal`` -> ``treeifyBin`` -> ``resize``).
    A key whose Java hash is not modelled, or a bin that would become a red-black tree (9 nodes at 64 or more
    buckets, iterated root first), keeps Python's order (a documented difference)."""
    items = list(m.items())
    spread: list[int] = []
    for k, _ in items:
        h = _java_hash(k)
        if h is None:
            return items
        spread.append(h ^ (h >> 16))
    size = 16
    bins: dict[int, int] = {}
    for count, s in enumerate(spread, start=1):
        bucket = s & (size - 1)
        bins[bucket] = bins.get(bucket, 0) + 1
        resized = False
        if bins[bucket] >= 9:
            if size >= 64:
                return items
            size *= 2
            resized = True
        if count > size * 3 // 4:
            size *= 2
            resized = True
        if resized:
            bins = {}
            for t in spread[:count]:
                bins[t & (size - 1)] = bins.get(t & (size - 1), 0) + 1
    return [items[i] for _, i in sorted((s & (size - 1), i) for i, s in enumerate(spread))]


def _java_text(v: Any) -> str:
    return "null" if v is None else _java_string(v)


def _java_double(v: float) -> str:
    """``Double.toString``: plain from 1e-3 up to 1e7, else ``d.dddE±n``."""
    import math

    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if v == 0:
        return "-0.0" if math.copysign(1.0, v) < 0 else "0.0"
    if 1e-3 <= abs(v) < 1e7:
        text = repr(v)
        return text if "." in text or "e" in text else text + ".0"
    import decimal

    _, digits, exponent = decimal.Decimal(repr(abs(v))).as_tuple()
    text = "".join(map(str, digits)).rstrip("0") or "0"
    power = len(digits) - 1 + int(exponent)
    mantissa = text[0] + "." + (text[1:] or "0")
    return f"{'-' if v < 0 else ''}{mantissa}E{power}"


def _from_java(dt: Any, v: Any) -> Any:
    """``EvaluatePython.makeFromJava(dt)`` (v4.2.0) of the value Pyrolite makes of ``v``, as the internal value
    (days for a date, microseconds for a timestamp) of the declared type; ``None`` where Spark's match fails."""
    import decimal

    from krishiv._udf.types import (
        ArrayType,
        BinaryType,
        BooleanType,
        ByteType,
        DateType,
        DayTimeIntervalType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        ShortType,
        StringType,
        StructType,
        TimestampNTZType,
        TimestampType,
        TimeType,
    )

    if v is None:
        return None
    kind = _java_kind(v)
    if isinstance(dt, BooleanType):
        return v if kind == "Boolean" else None
    for t, bits in ((ByteType, 8), (ShortType, 16), (IntegerType, 32), (LongType, 64)):
        if isinstance(dt, t):
            return _wrap(v, bits) if kind in ("Integer", "Long") else None
    if isinstance(dt, FloatType):
        if kind != "Double":
            return None
        import math
        import struct

        # Java's `Double.toFloat`: past the float range it is an infinity, not an error.
        if math.isfinite(v) and abs(v) >= 3.4028235677973366e38:
            return math.copysign(math.inf, v)
        return struct.unpack("f", struct.pack("f", v))[0]
    if isinstance(dt, DoubleType):
        return v if kind == "Double" else None
    if isinstance(dt, DecimalType):
        if kind != "BigDecimal":
            return None
        quantum = decimal.Decimal(1).scaleb(-dt.scale)
        rounded = v.quantize(quantum, rounding=decimal.ROUND_HALF_UP, context=decimal.Context(prec=100))
        if len(rounded.as_tuple().digits) > dt.precision and rounded != 0:
            raise SparkJvmError(
                "NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION",
                {
                    "roundedValue": str(rounded),
                    "originalValue": str(v),
                    "precision": str(dt.precision),
                    "scale": str(dt.scale),
                },
            )
        return rounded
    if isinstance(dt, DateType):
        return v if kind == "Integer" else None
    if isinstance(dt, (TimestampType, TimestampNTZType, DayTimeIntervalType, TimeType)):
        return v if kind in ("Integer", "Long") else None
    if isinstance(dt, StringType):
        return _java_string(v)
    if isinstance(dt, BinaryType):
        if kind == "String":
            return v.encode("utf-8")
        return bytes(v) if kind == "byte[]" else None
    if isinstance(dt, ArrayType):
        if kind in ("List", "Object[]"):
            return [_from_java(dt.elementType, e) for e in v]
        return None
    if isinstance(dt, MapType):
        if kind != "Map":
            return None
        return [(_from_java(dt.keyType, k), _from_java(dt.valueType, x)) for k, x in v.items()]
    if isinstance(dt, StructType):
        if kind != "Object[]":
            return None
        if len(v) != len(dt.fields):
            raise SparkJvmError(
                "STRUCT_ARRAY_LENGTH_MISMATCH", {"expected": str(len(dt.fields)), "actual": str(len(v))}
            )
        return {f.name: _from_java(f.dataType, x) for f, x in zip(dt.fields, v, strict=True)}
    return None


def _jvm_batch(rows: list[Any], return_type: StructType, arrow_return_type: Any) -> Any:
    """300's rows (each ``toInternal``'s result, or ``None`` for a ``None`` row) as the JVM keeps them, in Arrow:
    each field through :func:`_from_java`; a ``None`` row is all NULL (``EvalPythonUDTFExec``'s null row)."""
    import pyarrow as pa

    columns: list[list[Any]] = [[] for _ in return_type.fields]
    for row in rows:
        for i, f in enumerate(return_type.fields):
            columns[i].append(None if row is None else _from_java(f.dataType, row[i]))
    schema = pa.schema(list(arrow_return_type))
    return pa.RecordBatch.from_arrays(
        [pa.array(c, type=field.type) for c, field in zip(columns, schema, strict=True)], schema=schema
    )


def _check_table_udf_row(result: Any, return_type_size: int, func: str) -> None:
    """300's per-row checks of ``wrap_udtf``'s ``verify_and_convert_result`` (``worker.py:2233-2263``)."""
    if result is None:
        return
    if hasattr(result, "__UDT__"):
        raise PySparkRuntimeError(
            errorClass="UDTF_INVALID_OUTPUT_ROW_TYPE",
            messageParameters={"type": type(result).__name__, "func": func},
        )
    if hasattr(result, "__len__") and len(result) != return_type_size:
        raise PySparkRuntimeError(
            errorClass="UDTF_RETURN_SCHEMA_MISMATCH",
            messageParameters={"expected": str(return_type_size), "actual": str(len(result)), "func": func},
        )
    if not (isinstance(result, (list, dict, tuple)) or hasattr(result, "__dict__")):
        raise PySparkRuntimeError(
            errorClass="UDTF_INVALID_OUTPUT_ROW_TYPE",
            messageParameters={"type": type(result).__name__, "func": func},
        )


def _conversion_error(data: Any, return_type: StructType, arrow_return_type: Any) -> Exception:
    return PySparkRuntimeError(
        errorClass="UDTF_ARROW_DATA_CONVERSION_ERROR",
        messageParameters={
            "data": str(data),
            "schema": return_type.simpleString(),
            "arrow_schema": str(arrow_return_type),
        },
    )


def _arrow_udtf_mapper(
    udtf: Any,
    eval_type: int,
    return_type: StructType,
    call: UdtfCall,
    args_offsets: list[int],
    kwargs_offsets: dict[str, int],
    conf: RunnerConf,
    batches: Iterable[Any],
) -> Iterator[Any]:
    """302 (``worker.py:2137-2226``, with ``ArrowStreamArrowUDTFSerializer``'s load and its ``dump_stream``
    type coercion, ``serializers.py:256-300``): ``eval`` per input batch, a ``TABLE`` argument as a record
    batch and the others as arrays; each result batch coerced to the declared types."""
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowBatchTransformer
    from krishiv._udf.pandas_types import to_arrow_type

    skip = _skip_rest_exception()
    arrow_return_type = to_arrow_type(return_type, timezone="UTC", prefers_large_types=conf.use_large_var_types)
    return_type_size = len(return_type)
    target = pa.schema(list(arrow_return_type))

    def wrap(f: Callable[..., Any]) -> Callable[..., Iterator[Any]]:
        fname = getattr(f, "__name__", "")

        def func(*a: Any) -> Any:
            try:
                return f(*a)
            except skip:
                raise
            except Exception as e:
                raise PySparkRuntimeError(
                    errorClass="UDTF_EXEC_ERROR",
                    messageParameters={"method_name": fname, "error": str(e)},
                ) from None

        def check_return_value(res: Any) -> Any:
            if res is not None:
                if not isinstance(res, Iterable):
                    raise PySparkRuntimeError(
                        errorClass="UDTF_RETURN_NOT_ITERABLE",
                        messageParameters={"type": type(res).__name__, "func": fname},
                    )
                return res
            return iter([])

        def convert_to_arrow(data: Any) -> Iterator[Any]:
            is_empty = True
            for item in check_return_value(data):
                is_empty = False
                if isinstance(item, pa.Table):
                    yield from item.to_batches()
                elif isinstance(item, pa.RecordBatch):
                    yield item
                else:
                    raise PySparkRuntimeError(errorClass="UDTF_ARROW_TYPE_CONVERSION_ERROR", messageParameters={})
            if is_empty:
                yield pa.RecordBatch.from_pylist([], schema=target)

        def verify_result(result: Any) -> Any:
            if result.num_columns != return_type_size:
                raise PySparkRuntimeError(
                    errorClass="UDTF_RETURN_SCHEMA_MISMATCH",
                    messageParameters={
                        "expected": str(return_type_size),
                        "actual": str(result.num_columns),
                        "func": fname,
                    },
                )
            return result

        def evaluate(*args: Any) -> Iterator[Any]:
            for batch in convert_to_arrow(func(*args)):
                yield ArrowBatchTransformer.enforce_schema(verify_result(batch), target, safecheck=True)

        return evaluate

    eval_func, args_kwargs_offsets = _wrap_kwargs_support(udtf.eval, args_offsets, kwargs_offsets)
    evaluate = wrap(eval_func)
    terminate = wrap(udtf.terminate) if hasattr(udtf, "terminate") else None
    cleanup = udtf.cleanup if hasattr(udtf, "cleanup") else None
    try:
        for batch in batches:
            columns = [
                ArrowBatchTransformer.flatten_struct(batch, column_index=i)
                if i in call.table_arg_offsets
                else batch.column(i)
                for i in range(batch.num_columns)
            ]
            yield from evaluate(*[columns[o] for o in args_kwargs_offsets])
        if terminate is not None:
            yield from terminate()
    except skip:
        if terminate is not None:
            yield from terminate()
    finally:
        if cleanup is not None:
            cleanup()


def _dumps(value: Any) -> bytes:
    """``value`` pickled as Spark's ``pickleSer`` pickles it: with pyspark's ``cloudpickle`` in a pool worker."""
    if IN_WORKER:
        from pyspark import cloudpickle  # ty: ignore[unresolved-import]  # pool workers only

        return cloudpickle.dumps(value)
    import pickle

    return pickle.dumps(value)


def _analyze_api() -> tuple[Any, Any, Any, Any, Any]:
    """``AnalyzeArgument``, ``AnalyzeResult``, ``OrderingColumn``, ``PartitioningColumn``, ``SelectedColumn``:
    pyspark's in a pool worker (a UDTF's ``analyze`` builds them); the facade has none yet (PR-48)."""
    if not IN_WORKER:
        raise not_implemented("a Python UDTF's analyze in-process")
    from pyspark.sql.functions import (  # ty: ignore[unresolved-import]  # pool workers only
        OrderingColumn,
        PartitioningColumn,
        SelectedColumn,
    )
    from pyspark.sql.udtf import AnalyzeArgument, AnalyzeResult  # ty: ignore[unresolved-import]

    return AnalyzeArgument, AnalyzeResult, OrderingColumn, PartitioningColumn, SelectedColumn


def _internal_value(value: Any, dt: Any) -> Any:
    """Spark's internal form of the Arrow scalar ``value`` of Spark type ``dt``, as the JVM pickles a constant
    for ``analyze`` (``EvaluatePython.toJava`` at v4.2.0): a date as its days, a timestamp (either kind) or a
    day-time interval as its microseconds, a time as its nanoseconds, a struct as the tuple of its fields, an
    array as a list and a map as a dict of internal values; a decimal is a ``Decimal`` keeping its scale
    (``toJavaBigDecimal``), a binary is ``bytes`` (``spark.sql.execution.pyspark.binaryAsBytes``, true by
    default), any other value its plain Python value. ``dt.fromInternal`` then makes the Python value."""
    import pyarrow as pa

    if not value.is_valid:
        return None
    t = value.type
    if pa.types.is_date32(t):
        return value.cast(pa.int32()).as_py()
    if pa.types.is_timestamp(t) or pa.types.is_duration(t) or pa.types.is_time64(t):
        return value.cast(pa.int64()).as_py()
    if pa.types.is_struct(t) and hasattr(dt, "fields"):
        return tuple(_internal_value(value[i], f.dataType) for i, f in enumerate(dt.fields))
    if pa.types.is_map(t):
        return {_internal_value(e[0], dt.keyType): _internal_value(e[1], dt.valueType) for e in value.values}
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return [_internal_value(v, dt.elementType) for v in value.values]
    return value.as_py()


def analyze_udtf(handler: Any, name: str, args: list[dict[str, Any]], values: Any) -> dict[str, Any]:
    """Runs ``handler``'s static ``analyze`` for a call with ``args`` (each ``{type, is_table, name, column}``:
    a Spark JSON type, and for a constant the column of ``values``, a one-row record batch, holding its value)
    and returns its result as the server reads it
    (``analyze_udtf.py``'s ``_main``, v4.2.0): the schema (Spark JSON), the pickled ``AnalyzeResult`` (hex),
    ``withSinglePartition``, the ``partitionBy`` names, the ``orderBy`` columns and the ``select`` columns. Every
    failure raises; the server makes it ``TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON``."""
    from textwrap import dedent

    AnalyzeArgument, AnalyzeResult, OrderingColumn, PartitioningColumn, SelectedColumn = _analyze_api()

    # read_udtf (analyze_udtf.py:46-65).
    if not isinstance(handler, type):
        raise PySparkRuntimeError(
            f"Invalid UDTF handler type. Expected a class (type 'type'), but "
            f"got an instance of {type(handler).__name__}."
        )
    if not hasattr(handler, "analyze") or not isinstance(inspect.getattr_static(handler, "analyze"), staticmethod):
        raise PySparkRuntimeError(
            "Failed to execute the user defined table function because it has not "
            "implemented the 'analyze' static method or specified a fixed "
            "return type during registration time. "
            "Please add the 'analyze' static method or specify the return type, "
            "and try the query again."
        )
    # read_arguments (analyze_udtf.py:68-94).
    pos: list[Any] = []
    named: dict[str, Any] = {}
    for a in args:
        dt = _parse_type(str(a["type"]))
        constant = a.get("column") is not None
        value = None
        if constant:
            # analyze_udtf.py:79-82: the pickled internal value, through `fromInternal` when the type needs it.
            value = _internal_value(values.column(int(a["column"]))[0], dt)
            if dt.needConversion():
                value = dt.fromInternal(value)
        argument = AnalyzeArgument(
            dataType=dt,
            value=value,
            isTable=bool(a.get("is_table")),
            isConstantExpression=constant,
        )
        if a.get("name") is not None:
            named[str(a["name"])] = argument
        else:
            pos.append(argument)

    error_prefix = f"Failed to evaluate the user-defined table function '{name}'"

    def format_error(msg: str) -> str:
        return dedent(msg).replace("\n", " ")

    try:
        inspect.signature(handler.analyze).bind(*pos, **named)
    except TypeError as e:
        raise PySparkValueError(
            format_error(f"""
                {error_prefix} because the function arguments did not match the expected
                signature of the static 'analyze' method ({e}). Please update the query so that
                this table function call provides arguments matching the expected signature, or
                else update the table function so that its static 'analyze' method accepts the
                provided arguments, and then try the query again.""")
        ) from None

    result = handler.analyze(*pos, **named)

    if not isinstance(result, AnalyzeResult):
        raise PySparkValueError(
            format_error(f"""
                {error_prefix} because the static 'analyze' method expects a result of type
                pyspark.sql.udtf.AnalyzeResult, but instead this method returned a value of
                type: {type(result)}""")
        )
    if not isinstance(result.schema, StructType):
        raise PySparkValueError(
            format_error(f"""
                {error_prefix} because the static 'analyze' method expects a result of type
                pyspark.sql.udtf.AnalyzeResult with a 'schema' field comprising a StructType,
                but the 'schema' field had the wrong type: {type(result.schema)}""")
        )

    def invalid_analyze_result_field(field_name: str, expected_field: str) -> Exception:
        return PySparkValueError(
            format_error(f"""
                {error_prefix} because the static 'analyze' method returned an
                'AnalyzeResult' object with the '{field_name}' field set to a value besides a
                list or tuple of '{expected_field}' objects. Please update the table function
                and then try the query again.""")
        )

    has_table_arg = any(a.isTable for a in pos) or any(a.isTable for a in named.values())
    if not has_table_arg and result.withSinglePartition:
        raise PySparkValueError(
            format_error(f"""
                {error_prefix} because the static 'analyze' method returned an
                'AnalyzeResult' object with the 'withSinglePartition' field set to 'true', but
                the function call did not provide any table argument. Please update the query so
                that it provides a table argument, or else update the table function so that its
                'analyze' method returns an 'AnalyzeResult' object with the
                'withSinglePartition' field set to 'false', and then try the query again.""")
        )
    if not has_table_arg and len(result.partitionBy) > 0:
        raise PySparkValueError(
            format_error(f"""
                {error_prefix} because the static 'analyze' method returned an
                'AnalyzeResult' object with the 'partitionBy' list set to non-empty, but the
                function call did not provide any table argument. Please update the query so
                that it provides a table argument, or else update the table function so that its
                'analyze' method returns an 'AnalyzeResult' object with the 'partitionBy' list
                set to empty, and then try the query again.""")
        )
    if not isinstance(result.partitionBy, (list, tuple)) or not all(
        isinstance(val, PartitioningColumn) for val in result.partitionBy
    ):
        raise invalid_analyze_result_field("partitionBy", "PartitioningColumn")
    if not isinstance(result.orderBy, (list, tuple)) or not all(
        isinstance(val, OrderingColumn) for val in result.orderBy
    ):
        raise invalid_analyze_result_field("orderBy", "OrderingColumn")
    if not isinstance(result.select, (list, tuple)) or not all(
        isinstance(val, SelectedColumn) for val in result.select
    ):
        raise invalid_analyze_result_field("select", "SelectedColumn")

    return {
        "schema": result.schema.json(),
        # analyze_udtf.py:203 writes it with `pickleSer` (cloudpickle), so a result class made in the client's
        # `__main__` pickles too.
        "pickled": _dumps(result).hex(),
        "with_single_partition": bool(result.withSinglePartition),
        "partition_by": [c.name for c in result.partitionBy],
        "order_by": [
            {"name": c.name, "ascending": bool(c.ascending), "nulls_first": c.overrideNullsFirst}
            for c in result.orderBy
        ],
        "select": [{"name": c.name, "alias": c.alias} for c in result.select],
    }


__all__ = [
    "SQL_ARROW_TABLE_UDF",
    "SQL_ARROW_UDTF",
    "SQL_TABLE_UDF",
    "UDTF_EVAL_TYPES",
    "PySparkTypeError",
    "UdtfCall",
    "analyze_udtf",
    "make_udtf_runner",
]
