# SPDX-License-Identifier: Apache-2.0
# Parts ported from Apache Spark v4.2.0, python/pyspark/worker.py, python/pyspark/util.py and
# python/pyspark/sql/pandas/serializers.py (Apache License 2.0; see NOTICE). Each ported function names its
# source lines; the code keeps Spark's shape so a later diff against the tag stays readable.
"""The wrappers the embedded runtime calls (design P2-T059 §2.5, unit 2c): one per scalar eval type, each
``wrapper(*pyarrow.Array) -> pyarrow.Array``, one argument array per UDF argument and one result array.

Spark's worker reads one Arrow stream per task and runs a chain of UDFs over each batch; here the engine
calls one UDF per batch, so each wrapper is the body of Spark's per-batch loop for one UDF:

- ``SQL_ARROW_BATCHED_UDF`` (101, ``@udf``): ``worker.py:2999-3090``, the non-legacy path, and with
  ``spark.sql.legacy.execution.pythonUDF.pandas.conversion.enabled`` the legacy one through pandas
  (``worker.py:3093-3180``).
- ``SQL_SCALAR_PANDAS_UDF`` (200, scalar ``pandas_udf``): ``wrap_scalar_pandas_udf`` (``worker.py:381-414``)
  between ``ArrowStreamPandasUDFSerializer``'s load and dump (``serializers.py:363-549``).
- ``SQL_SCALAR_ARROW_UDF`` (250, ``arrow_udf``): ``worker.py:2553-2582``, no conversion.

Two steps are Krishiv's own, around Spark's: the engine passes strings and binaries as Arrow views, which
are cast to Spark's ``string`` / ``binary`` before the conversion, and the result, built with Spark's Arrow
type, is cast to the engine's (views again) after it. Both casts are lossless.

The runner conf (the SQL conf Spark sends its worker, ``worker.py:113-184``) comes from the session the call
runs in: the runtime hands it to the running call (``_native._udf_runner_conf()``), so one wrapper serves
every session.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from krishiv.errors import PySparkRuntimeError, PySparkTypeError
from krishiv.sql.types import DataType, StructField, StructType

#: Spark's ``PythonEvalType`` codes (``pyspark/util.py:646-688``) of the kinds the engine runs.
SQL_BATCHED_UDF = 100
SQL_ARROW_BATCHED_UDF = 101
SQL_SCALAR_PANDAS_UDF = 200
SQL_SCALAR_ARROW_UDF = 250


class RunnerConf:
    """The subset of ``worker.py:113-184`` ``RunnerConf`` these eval types read, with Spark's defaults
    (``Conf.get``, ``worker_util.py:250-254``: string values lower-cased unless ``lower_str=False``)."""

    def __init__(self, conf: dict[str, str]) -> None:
        self._conf = conf

    def get(self, key: str, default: Any = "", *, lower_str: bool = True) -> Any:
        val = self._conf.get(key, default)
        if isinstance(val, str) and lower_str:
            return val.lower()
        return val

    @property
    def assign_cols_by_name(self) -> bool:
        return self.get("spark.sql.legacy.execution.pandas.groupedMap.assignColumnsByName", "true") == "true"

    @property
    def use_large_var_types(self) -> bool:
        return self.get("spark.sql.execution.arrow.useLargeVarTypes", "false") == "true"

    @property
    def binary_as_bytes(self) -> bool:
        return self.get("spark.sql.execution.pyspark.binaryAsBytes", "true") == "true"

    @property
    def safecheck(self) -> bool:
        return self.get("spark.sql.execution.pandas.convertToArrowArraySafely", "false") == "true"

    @property
    def int_to_decimal_coercion_enabled(self) -> bool:
        return self.get("spark.sql.execution.pythonUDF.pandas.intToDecimalCoercionEnabled", "false") == "true"

    @property
    def prefer_int_ext_dtype(self) -> bool:
        return self.get("spark.sql.execution.pythonUDF.pandas.preferIntExtensionDtype", "false") == "true"

    @property
    def timezone(self) -> str | None:
        return self.get("spark.sql.session.timeZone", None, lower_str=False)

    @property
    def use_legacy_pandas_udf_conversion(self) -> bool:
        return self.get("spark.sql.legacy.execution.pythonUDF.pandas.conversion.enabled", "false") == "true"

    @property
    def arrow_concurrency_level(self) -> int:
        return int(self.get("spark.sql.execution.pythonUDF.arrow.concurrency.level", -1))


def _runner_conf() -> RunnerConf:
    from krishiv import _native

    return RunnerConf(_native._udf_runner_conf())


# --- ported from pyspark/util.py:151-168 ---
def fail_on_stopiteration(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wraps the input function to fail on 'StopIteration' by raising a 'RuntimeError'
    prevents silent loss of data when 'f' is used in a for loop in Spark code
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return f(*args, **kwargs)
        except StopIteration as exc:
            raise PySparkRuntimeError(
                errorClass="STOP_ITERATION_OCCURRED",
                messageParameters={
                    "exc": str(exc),
                },
            ) from exc

    return wrapper


# --- ported from pyspark/worker.py:279-288 ---
def verify_result_row_count(result_length: int, expected: int) -> None:
    """Raise if the result row count doesn't match the expected input row count."""
    if result_length != expected:
        raise PySparkRuntimeError(
            errorClass="RESULT_ROWS_MISMATCH",
            messageParameters={
                "output_length": str(result_length),
                "input_length": str(expected),
            },
        )


# --- ported from pyspark/worker.py:291-322 ---
def verify_scalar_result(result: Any, num_rows: int) -> Any:
    """
    Verify a scalar UDF result is array-like and has the expected number of rows.
    """
    try:
        result_length = len(result)
    except TypeError:
        raise PySparkTypeError(
            errorClass="UDF_RETURN_TYPE",
            messageParameters={
                "expected": "array-like object",
                "actual": type(result).__name__,
            },
        ) from None
    if result_length != num_rows:
        raise PySparkRuntimeError(
            errorClass="SCHEMA_MISMATCH_FOR_PANDAS_UDF",
            messageParameters={
                "udf_type": "arrow_udf",
                "expected": str(num_rows),
                "actual": str(result_length),
            },
        )
    return result


# --- ported from pyspark/worker.py:381-414 ---
def wrap_scalar_pandas_udf(f: Callable[..., Any], return_type: DataType) -> Callable[..., Any]:
    def verify_result_type(result: Any) -> Any:
        if not hasattr(result, "__len__"):
            pd_type = "pandas.DataFrame" if isinstance(return_type, StructType) else "pandas.Series"
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={
                    "expected": pd_type,
                    "actual": type(result).__name__,
                },
            )
        return result

    def verify_result_length(result: Any, length: int) -> Any:
        if len(result) != length:
            raise PySparkRuntimeError(
                errorClass="SCHEMA_MISMATCH_FOR_PANDAS_UDF",
                messageParameters={
                    "udf_type": "pandas_udf",
                    "expected": str(length),
                    "actual": str(len(result)),
                },
            )
        return result

    return lambda *a: (
        verify_result_length(verify_result_type(f(*a)), len(a[0])),
        return_type,
    )


# --- Krishiv: the engine's Arrow forms around Spark's ---
def _spark_arrow_type(t: Any, timezone: str | None = None) -> Any:
    """``t`` with the engine's view types replaced by Spark's (``string``, ``binary``) and, given ``timezone``,
    its zoned timestamps in that zone (Spark sends the session time zone; the engine sends UTC: the same instants)."""
    import pyarrow as pa

    if timezone is not None and pa.types.is_timestamp(t) and t.tz is not None:
        return pa.timestamp(t.unit, tz=timezone)
    if pa.types.is_string_view(t):
        return pa.string()
    if pa.types.is_binary_view(t):
        return pa.binary()
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        field = t.value_field
        inner = field.with_type(_spark_arrow_type(field.type, timezone))
        return pa.list_(inner) if pa.types.is_list(t) else pa.large_list(inner)
    if pa.types.is_map(t):
        return pa.map_(
            t.key_field.with_type(_spark_arrow_type(t.key_type, timezone)),
            t.item_field.with_type(_spark_arrow_type(t.item_type, timezone)),
        )
    if pa.types.is_struct(t):
        return pa.struct([f.with_type(_spark_arrow_type(f.type, timezone)) for f in t])
    return t


def _engine_arrow_type(t: Any) -> Any:
    """Spark's Arrow type ``t`` as the engine declares results: strings and binaries as views."""
    import pyarrow as pa

    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return pa.string_view()
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return pa.binary_view()
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        field = t.value_field
        return pa.list_(field.with_type(_engine_arrow_type(field.type)))
    if pa.types.is_map(t):
        return pa.map_(
            t.key_field.with_type(_engine_arrow_type(t.key_type)),
            t.item_field.with_type(_engine_arrow_type(t.item_type)),
        )
    if pa.types.is_struct(t):
        return pa.struct([f.with_type(_engine_arrow_type(f.type)) for f in t])
    return t


def _to_spark(arr: Any, timezone: str | None = None) -> Any:
    t = _spark_arrow_type(arr.type, timezone)
    return arr if t == arr.type else arr.cast(t)


def _to_engine(arr: Any, engine_type: Any) -> Any:
    return arr if arr.type == engine_type else arr.cast(engine_type)


def _arrow_return_type(return_type: DataType, conf: RunnerConf) -> Any:
    from krishiv._udf.pandas_types import to_arrow_type

    return to_arrow_type(return_type, timezone="UTC", prefers_large_types=conf.use_large_var_types)


# --- SQL_ARROW_BATCHED_UDF (101): pyspark/worker.py:2999-3090, one UDF ---
def arrow_batched_wrapper(
    udf_func: Callable[..., Any], return_type: DataType, *, legacy_allowed: bool = False
) -> Callable[..., Any]:
    """``@udf``: rows in, ``udf_func`` per row, rows out (the non-legacy path, ``worker.py:2999-3090``; with
    ``legacy_allowed`` (eval type 101) and the legacy conf set, ``worker.py:3093-3180``)."""
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowTableToRowsConversion, LocalDataToArrowConversion
    from krishiv._udf.pandas_types import from_arrow_type
    from krishiv._udf.types import _NoValue

    # `read_single_udf` does not wrap this eval type in `fail_on_stopiteration` (`worker.py:1103-1110`); the
    # batch evaluation is (`worker.py:3040`).
    @fail_on_stopiteration
    def _evaluate_batch_udf(udf_func: Callable[..., Any], rows: list[Any], conf: RunnerConf) -> list[Any]:
        if conf.arrow_concurrency_level <= 0:
            return [udf_func(*row) for row in rows]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=conf.arrow_concurrency_level) as pool:
            return list(pool.map(lambda row: udf_func(*row), rows))

    legacy = _arrow_batched_legacy(udf_func, return_type) if legacy_allowed else None

    def func(*arrays: Any) -> Any:
        conf = _runner_conf()
        if legacy is not None and conf.use_legacy_pandas_udf_conversion:
            return legacy(conf, *arrays)
        zero_arg = len(arrays) == 0
        arrow_return_type = _arrow_return_type(return_type, conf)
        result_conv = LocalDataToArrowConversion._create_converter(
            return_type,
            none_on_identity=True,
            int_to_decimal_coercion_enabled=conf.int_to_decimal_coercion_enabled,
        )
        columns_in = [_to_spark(a) for a in arrays]
        # --- Input preparation --- (the engine sends no input schema: the types come from the arrays)
        arrow_to_py_converters = [
            ArrowTableToRowsConversion._create_converter(
                from_arrow_type(col.type), none_on_identity=True, binary_as_bytes=conf.binary_as_bytes
            )
            for col in columns_in
        ]
        num_rows = len(arrays[0]) if arrays else _zero_arg_rows()

        # --- Input: Arrow -> Python columns ---
        columns = [
            (
                [conv(v) for v in ArrowTableToRowsConversion._to_pylist(col)]
                if conv is not None
                else ArrowTableToRowsConversion._to_pylist(col)
            )
            for col, conv in zip(columns_in, arrow_to_py_converters, strict=True)
        ]
        if not columns:
            columns = [[_NoValue] * num_rows]

        # --- Process: evaluate the UDF row-by-row ---
        rows = [() for _ in range(num_rows)] if zero_arg else list(zip(*columns, strict=True))
        results = _evaluate_batch_udf(udf_func, rows, conf)
        verify_result_row_count(len(results), num_rows)

        # --- Output: Python -> Arrow ---
        converted = [result_conv(r) for r in results] if result_conv is not None else results
        try:
            arr = pa.array(converted, type=arrow_return_type)
        except pa.lib.ArrowInvalid:
            arr = pa.array(converted).cast(target_type=arrow_return_type, safe=conf.safecheck)
        return _to_engine(arr, _engine_arrow_type(arrow_return_type))

    return func


# --- SQL_ARROW_BATCHED_UDF (101), legacy conversion: pyspark/worker.py:3093-3180, one UDF ---
def _arrow_batched_legacy(udf_func: Callable[..., Any], return_type: DataType) -> Callable[..., Any]:
    """The legacy ``@udf`` path: Arrow -> pandas columns -> rows, ``udf_func`` per row, pandas -> Arrow, with
    string and binary results coerced to ``str`` / ``bytes`` (binary arguments arrive as ``bytes`` whatever
    ``binaryAsBytes`` says)."""
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowBatchTransformer, PandasToArrowConversion
    from krishiv._udf.types import _NoValue
    from krishiv.sql.types import BinaryType, StringType

    # Legacy coerces String/Binary for Arrow compatibility
    coerce = str if isinstance(return_type, StringType) else bytes if isinstance(return_type, BinaryType) else None
    return_schema = StructType([StructField("_0", return_type)])

    @fail_on_stopiteration
    def _evaluate_batch_udf_legacy(udf_func: Callable[..., Any], rows: list[Any], conf: RunnerConf) -> list[Any]:
        if conf.arrow_concurrency_level <= 0:
            return [udf_func(*row) for row in rows]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=conf.arrow_concurrency_level) as pool:
            return list(pool.map(lambda row: udf_func(*row), rows))

    def func(conf: RunnerConf, *arrays: Any) -> Any:
        import pandas as pd

        zero_arg = len(arrays) == 0
        num_rows = len(arrays[0]) if arrays else _zero_arg_rows()
        input_batch = pa.RecordBatch.from_arrays(
            [_to_spark(a) for a in arrays], names=[f"_{i}" for i in range(len(arrays))]
        )
        # --- Input: Arrow -> pandas columns --- (the engine sends no input schema: the types come from the
        # arrays, as `to_pandas` reads them without one)
        pandas_columns = ArrowBatchTransformer.to_pandas(
            input_batch,
            timezone=conf.timezone,  # ty: ignore[invalid-argument-type]  # Optional, as Spark passes it
            struct_in_pandas="row",
            ndarray_as_list=True,
            prefer_int_ext_dtype=conf.prefer_int_ext_dtype,
            df_for_struct=False,
        )
        if not pandas_columns:
            pandas_columns = [pd.Series([_NoValue] * num_rows)]

        # --- Process: evaluate the UDF row-by-row ---
        rows = (
            [() for _ in range(num_rows)] if zero_arg else list(zip(*[c.tolist() for c in pandas_columns], strict=True))
        )
        results = _evaluate_batch_udf_legacy(udf_func, rows, conf)
        verify_result_row_count(len(results), num_rows)
        if coerce:
            results = [coerce(v) if v is not None else v for v in results]

        # --- Output: pandas -> Arrow ---
        out = PandasToArrowConversion.convert(
            [pd.Series(results)],
            return_schema,
            timezone=conf.timezone,
            safecheck=conf.safecheck,
            arrow_cast=True,
            prefers_large_types=conf.use_large_var_types,
            assign_cols_by_name=conf.assign_cols_by_name,
            int_to_decimal_coercion_enabled=conf.int_to_decimal_coercion_enabled,
        ).column(0)
        return _to_engine(out, _engine_arrow_type(_arrow_return_type(return_type, conf)))

    return func


def _zero_arg_rows() -> int:
    """The rows of a call without arguments: the runtime passes none (the engine sends the batch's row count
    only with arguments), so it is read from the runner conf the runtime sets."""
    from krishiv import _native

    return int(_native._udf_runner_conf().get("krishiv.python.udf.rows", "0"))


# --- SQL_SCALAR_PANDAS_UDF (200): serializers.py:363-549 around worker.py:381-414 ---
def scalar_pandas_wrapper(f: Callable[..., Any], return_type: DataType) -> Callable[..., Any]:
    """Scalar ``pandas_udf``: Series (a DataFrame for a struct) in, a Series out."""
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowBatchTransformer, PandasToArrowConversion

    # `read_single_udf` wraps every eval type but 101 and the iterator kinds (`worker.py:1103-1110`).
    udf = wrap_scalar_pandas_udf(fail_on_stopiteration(f), return_type)

    def func(*arrays: Any) -> Any:
        import pandas as pd

        from krishiv.errors import PySparkValueError

        conf = _runner_conf()
        batch = pa.RecordBatch.from_arrays(
            [_to_spark(a, conf.timezone) for a in arrays], names=[f"_{i}" for i in range(len(arrays))]
        )
        # load_stream (`serializers.py:441-455`), with the worker's options (`worker.py:2467-2490`).
        series = ArrowBatchTransformer.to_pandas(
            batch,
            timezone=conf.timezone,  # ty: ignore[invalid-argument-type]  # Optional, as Spark passes it
            struct_in_pandas="dict",
            ndarray_as_list=False,
            prefer_int_ext_dtype=conf.prefer_int_ext_dtype,
            df_for_struct=True,
        )
        # The mapper (`worker.py:3543-3551`) for one UDF over every column.
        result, spark_type = udf(*series)
        # dump_stream's create_batch (`serializers.py:509-539`).
        if isinstance(spark_type, StructType) and not isinstance(result, pd.DataFrame):
            raise PySparkValueError(
                "Invalid return type. Please make sure that the UDF returns a "
                "pandas.DataFrame when the specified return type is StructType."
            )
        out = PandasToArrowConversion.convert(
            [result],
            StructType([StructField("_0", spark_type)]),
            timezone=conf.timezone,
            safecheck=conf.safecheck,
            arrow_cast=True,
            prefers_large_types=conf.use_large_var_types,
            assign_cols_by_name=conf.assign_cols_by_name,
            int_to_decimal_coercion_enabled=conf.int_to_decimal_coercion_enabled,
            ignore_unexpected_complex_type_values=False,
            is_legacy=False,
        ).column(0)
        return _to_engine(out, _engine_arrow_type(_arrow_return_type(return_type, conf)))

    return func


# --- SQL_SCALAR_ARROW_UDF (250): worker.py:2553-2582, one UDF ---
def scalar_arrow_wrapper(f: Callable[..., Any], return_type: DataType) -> Callable[..., Any]:
    """``arrow_udf``: the arrays as they are in, a ``pyarrow.Array`` out (no conversion)."""
    import pyarrow as pa

    from krishiv._udf.conversion import ArrowBatchTransformer

    udf_func = fail_on_stopiteration(f)

    def func(*arrays: Any) -> Any:
        conf = _runner_conf()
        result = udf_func(*[_to_spark(a, conf.timezone) for a in arrays])
        # Only pyarrow's own arrays are imported (design "Trusted producer"): anything else, even with
        # `__arrow_c_array__`, is refused here. Spark would fail in `RecordBatch.from_arrays`.
        if not isinstance(result, pa.Array):
            raise PySparkTypeError(
                errorClass="UDF_RETURN_TYPE",
                messageParameters={"expected": "pyarrow.Array", "actual": type(result).__name__},
            )
        arrow_return_type = _arrow_return_type(return_type, conf)
        output_batch = pa.RecordBatch.from_arrays([result], ["_0"])
        output_batch = ArrowBatchTransformer.enforce_schema(
            output_batch, pa.schema([pa.field("_0", arrow_return_type)])
        )
        num_rows = len(arrays[0]) if arrays else _zero_arg_rows()
        verify_scalar_result(output_batch, num_rows)
        return _to_engine(output_batch.column(0), _engine_arrow_type(arrow_return_type))

    return func


def make_wrapper(f: Callable[..., Any], eval_type: int, return_type: DataType) -> Callable[..., Any]:
    """The wrapper of ``f`` for ``eval_type``; ``SQL_BATCHED_UDF`` (100) runs on the Arrow-batched path
    (design owner question 2: the pickled-row path is not implemented, a documented divergence)."""
    if eval_type in (SQL_BATCHED_UDF, SQL_ARROW_BATCHED_UDF):
        return arrow_batched_wrapper(f, return_type, legacy_allowed=eval_type == SQL_ARROW_BATCHED_UDF)
    if eval_type == SQL_SCALAR_PANDAS_UDF:
        return scalar_pandas_wrapper(f, return_type)
    if eval_type == SQL_SCALAR_ARROW_UDF:
        return scalar_arrow_wrapper(f, return_type)
    from krishiv.errors import not_implemented

    raise not_implemented(f"Python UDFs of eval type {eval_type}")
