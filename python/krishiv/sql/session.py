# SPDX-License-Identifier: Apache-2.0
"""``SparkSession``: the entry point, as ``pyspark.sql.SparkSession`` (embedded: no server)."""

from __future__ import annotations

import threading
import warnings
from typing import Any

from krishiv import _native
from krishiv.errors import PySparkValueError, not_expected_type, not_implemented
from krishiv.sql.conf import RuntimeConfig
from krishiv.sql.dataframe import DataFrame
from krishiv.sql.readwriter import DataFrameReader

#: The Spark version this engine is compatible with.
SPARK_VERSION = "4.2.0"


def _arrow_type_of(data_type: Any) -> Any:
    """The pyarrow type of a flat Spark ``DataType`` (nested types need engine support)."""
    import pyarrow as pa

    from krishiv.sql import types as T

    mapping: dict[type, Any] = {
        T.NullType: pa.null(),
        T.BooleanType: pa.bool_(),
        T.ByteType: pa.int8(),
        T.ShortType: pa.int16(),
        T.IntegerType: pa.int32(),
        T.LongType: pa.int64(),
        T.FloatType: pa.float32(),
        T.DoubleType: pa.float64(),
        T.StringType: pa.string(),
        T.BinaryType: pa.binary(),
        T.DateType: pa.date32(),
        T.TimestampType: pa.timestamp("us", "UTC"),
        T.TimestampNTZType: pa.timestamp("us"),
    }
    if isinstance(data_type, T.DecimalType):
        return pa.decimal128(data_type.precision, data_type.scale)
    if (arrow := mapping.get(type(data_type))) is not None:
        return arrow
    raise not_implemented(f"createDataFrame with {type(data_type).__name__}")


_lock = threading.Lock()
_active: SparkSession | None = None


class SparkSession:
    """An embedded session. Create one with ``SparkSession.builder.getOrCreate()``."""

    class Builder:
        def __init__(self) -> None:
            self._config: dict[str, str] = {}

        def config(
            self,
            key: str | None = None,
            value: Any = None,
            *,
            conf: None = None,
            map: dict[str, Any] | None = None,
        ) -> SparkSession.Builder:
            if map is not None:
                for k, v in map.items():
                    self.config(k, v)
            if key is not None:
                self._config[key] = ("true" if value else "false") if isinstance(value, bool) else str(value)
            return self

        def appName(self, name: str) -> SparkSession.Builder:
            return self.config("spark.app.name", name)

        def master(self, master: str) -> SparkSession.Builder:
            return self.config("spark.master", master)

        def remote(self, url: str) -> SparkSession.Builder:
            raise not_implemented("remote sessions")

        def getOrCreate(self) -> SparkSession:
            """The active session (with this builder's options applied to it), or a new one."""
            global _active
            with _lock:
                if _active is None:
                    _active = SparkSession(_native.Session(list(self._config.items())))
                else:
                    skipped = [k for k in self._config if not _active._native.conf_is_modifiable(k)]
                    if skipped:
                        warnings.warn(
                            "Using an existing Spark session; only runtime SQL configurations will take effect "
                            f"(ignored: {', '.join(skipped)}).",
                            UserWarning,
                            stacklevel=2,
                        )
                    for k, v in self._config.items():
                        if k not in skipped:
                            _active.conf.set(k, v)
                return _active

        def create(self) -> SparkSession:
            """A new session, which does not become the active one."""
            return SparkSession(_native.Session(list(self._config.items())))

    class _BuilderProperty:
        """``SparkSession.builder``: a fresh Builder per access, so options do not leak between callers."""

        def __get__(self, instance: object, owner: type | None = None) -> SparkSession.Builder:
            return SparkSession.Builder()

    # Typed as Builder (what attribute access gives); the descriptor below provides it.
    builder: Builder = _BuilderProperty()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    def __init__(self, native: _native.Session) -> None:
        self._native = native
        self._conf = RuntimeConfig(native)

    @property
    def version(self) -> str:
        return SPARK_VERSION

    @property
    def conf(self) -> RuntimeConfig:
        return self._conf

    @property
    def read(self) -> DataFrameReader:
        return DataFrameReader(self)

    def sql(self, sqlQuery: str, args: Any = None, **kwargs: Any) -> DataFrame:
        """Parses and analyzes ``sqlQuery`` now."""
        if not isinstance(sqlQuery, str):
            raise not_expected_type("sqlQuery", "str", sqlQuery)
        if args is not None or kwargs:
            raise not_implemented("sql with parameters")
        return DataFrame(self._native.sql(sqlQuery), self)

    def table(self, tableName: str) -> DataFrame:
        return DataFrame(self._native.table(tableName), self)

    def createDataFrame(self, data: Any, schema: Any = None) -> DataFrame:
        """A DataFrame of Arrow data: any object with ``__arrow_c_stream__`` or ``__arrow_c_array__``
        (a pyarrow Table, a Polars or pandas-via-pyarrow frame, a DuckDB result, ...), or a list of
        rows (``Row``, tuples, lists or dicts) with a schema of column names or a ``StructType``."""
        if hasattr(data, "__arrow_c_stream__") or hasattr(data, "__arrow_c_array__"):
            if schema is not None:
                raise not_implemented("createDataFrame with a schema")
            return DataFrame(self._native.create_dataframe(data), self)
        if isinstance(data, list | tuple):
            return self._create_from_rows(list(data), schema)
        raise not_implemented(f"createDataFrame from {type(data).__name__}")

    def _create_from_rows(self, data: list[Any], schema: Any) -> DataFrame:
        """A DataFrame of Python rows, via pyarrow (flat ``StructType`` or name lists only)."""
        from krishiv.sql import _convert
        from krishiv.sql.row import Row
        from krishiv.sql.types import StructType

        pa = _convert.require_pyarrow("createDataFrame")
        if isinstance(schema, str):
            raise not_implemented("createDataFrame with a string schema")
        if schema is not None and not isinstance(schema, StructType | list | tuple):
            raise not_expected_type("schema", "StructType, list of names or None", schema)
        names: list[str] | None = None
        field_types: list[Any] | None = None
        if isinstance(schema, StructType):
            names = [f.name for f in schema]
            field_types = [f.dataType for f in schema]
        elif isinstance(schema, list | tuple):
            names = list(schema)
            for name in names:
                if not isinstance(name, str):
                    raise not_expected_type("schema", "list of str", name)
        for row in data:
            if not isinstance(row, Row | tuple | list | dict):
                raise not_expected_type("data", "list of Row, tuple, list or dict", row)
        if names is None:
            if not data:
                raise PySparkValueError("can not infer schema from empty data")
            first = data[0]
            if isinstance(first, Row) and hasattr(first, "__fields__"):
                names = list(first.__fields__)
            elif isinstance(first, dict):
                names = list(first)
            else:
                names = [f"_{i + 1}" for i in range(len(first))]
        width = len(names)
        per_column: list[list[Any]] = [[] for _ in names]
        for row in data:
            if isinstance(row, dict):
                for index, name in enumerate(names):
                    per_column[index].append(row.get(name))
            else:
                if len(row) != width:
                    raise PySparkValueError(f"row has {len(row)} values, expected {width}")
                for index in range(width):
                    per_column[index].append(row[index])
        columns: list[Any] = []
        for index, values in enumerate(per_column):
            if field_types is not None:
                columns.append(pa.array(values, type=_arrow_type_of(field_types[index])))
            elif data:
                inferred = pa.array(values)
                if pa.types.is_timestamp(inferred.type) and inferred.type.tz is None:
                    from datetime import UTC, datetime

                    if any(isinstance(v, datetime) for v in values if v is not None):
                        # PySpark interprets naive datetimes in the Python process's local
                        # time zone; attach it before storing the instant as UTC.
                        localized = [
                            v.astimezone(UTC) if isinstance(v, datetime) and v.tzinfo is None else v for v in values
                        ]
                        inferred = pa.array(localized, type=pa.timestamp(inferred.type.unit, "UTC"))
                columns.append(inferred)
            else:
                columns.append(pa.array([], type=pa.null()))
        if isinstance(schema, StructType):
            fields = [
                pa.field(name, _arrow_type_of(f.dataType), nullable=f.nullable)
                for name, f in zip(names, schema, strict=True)
            ]
            table = pa.table(columns, schema=pa.schema(fields))
        else:
            table = pa.table(columns, names=names)
        return DataFrame(self._native.create_dataframe(table), self)

    def stop(self) -> None:
        """Ends this session; the next ``getOrCreate`` makes a new one."""
        global _active
        with _lock:
            if _active is self:
                _active = None
