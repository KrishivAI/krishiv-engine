# SPDX-License-Identifier: Apache-2.0
"""The Python Data Source API (batch), shaped like ``pyspark.sql.datasource`` of Spark 4.3.0.

A user subclasses :class:`DataSource` (and :class:`DataSourceReader`, :class:`DataSourceWriter` or
:class:`DataSourceArrowWriter`), registers the class with ``spark.dataSource.register(cls)`` and then
reads with ``spark.read.format(name)`` and writes with ``df.write.format(name)``.

How it runs here:

* The schema is fixed when ``load`` is called (Spark's ``create_data_source`` step).
* The reader's ``partitions()`` and ``read()`` run when the query's scan runs (planning and ``EXPLAIN`` read nothing),
  once per scan, so
  a source that lists files sees new ones on the next action (Spark plans the read per execution).
  Partitions run one after the other in this process (the process pool's parallel partitions are T014).
* ``read`` may yield tuples, ``Row`` objects, lists or ``pyarrow.RecordBatch`` objects; they are cast to
  the declared schema.
* A write runs as one task over every row of the DataFrame, then ``commit`` (or ``abort`` when the task
  failed, with ``None`` for the failed task's message).
* Filter pushdown: ``spark.sql.python.filterPushdown.enabled`` is off by default, as in Spark 4.3.0; a
  reader that overrides ``pushFilters`` then fails with ``DATA_SOURCE_PUSHDOWN_DISABLED``. With the
  switch on, ``pushFilters`` is **not called** yet (the engine applies every filter itself); the pushdown
  acceptance of this API stays open.
* Interim embedded path: the read is materialised in memory when a scan runs (capped by
  ``KRISHIV_PYTHON_DATA_SOURCE_LIMIT``), not streamed per partition as the design's read operator will.
* The ``DataSource`` instance is created once by ``load`` and reused by every plan of the DataFrame (Spark
  re-creates it from the pickled class for each planning), so state a reader keeps on it lasts across actions.
* The streaming classes (``DataSourceStreamReader``, ``SimpleDataSourceStreamReader``,
  ``DataSourceStreamWriter``, ``DataSourceStreamArrowWriter``) are P4: ``streamReader`` and
  ``streamWriter`` raise ``NOT_IMPLEMENTED``.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from krishiv.errors import (
    AnalysisException,
    PySparkAssertionError,
    PySparkNotImplementedError,
    PySparkRuntimeError,
    PySparkTypeError,
    SparkRuntimeException,
)
from krishiv.sql.types import DataType, StructType

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame
    from krishiv.sql.session import SparkSession

__all__ = [
    "CaseInsensitiveDict",
    "DataSource",
    "DataSourceArrowWriter",
    "DataSourceReader",
    "DataSourceRegistration",
    "DataSourceWriter",
    "EqualNullSafe",
    "EqualTo",
    "Filter",
    "GreaterThan",
    "GreaterThanOrEqual",
    "In",
    "InputPartition",
    "IsNotNull",
    "IsNull",
    "LessThan",
    "LessThanOrEqual",
    "Not",
    "StringContains",
    "StringEndsWith",
    "StringStartsWith",
    "WriterCommitMessage",
]

#: The conf that lets the engine call ``DataSourceReader.pushFilters`` (internal in Spark 4.3.0, off by default).
PUSHDOWN_CONF = "spark.sql.python.filterPushdown.enabled"
#: Rows per Arrow batch a read hands the engine (Spark's ``spark.sql.execution.arrow.maxRecordsPerBatch``).
MAX_BATCH_ROWS = 10_000
#: The most bytes one plan of a source may hold (the interim embedded read path).
MAX_SOURCE_BYTES = 1 << 30

ColumnPath = tuple[str, ...]


class CaseInsensitiveDict(dict):  # type: ignore[type-arg]
    """The options of a data source: ``options["Path"]`` and ``options["path"]`` are one key."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key.lower(), value)

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key.lower())

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key.lower())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and super().__contains__(key.lower())

    def get(self, key: str, default: Any = None) -> Any:  # ty: ignore[invalid-method-override]
        return super().get(key.lower(), default)

    def pop(self, key: str, *default: Any) -> Any:  # ty: ignore[invalid-method-override]
        return super().pop(key.lower(), *default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return super().setdefault(key.lower(), default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        # Pairs are applied in order, so of ``Path`` then ``path`` the later one wins.
        for arg in args:
            for key, value in arg.items() if hasattr(arg, "items") else arg:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def copy(self) -> CaseInsensitiveDict:
        return CaseInsensitiveDict(self)


# ---- the filters ``pushFilters`` receives (Spark 4.1+) ----------------------------------------------


@dataclass(frozen=True)
class Filter(ABC):  # noqa: B024  (an abstract marker, as in Spark)
    """The base class of the filters a reader may be offered."""


@dataclass(frozen=True)
class EqualTo(Filter):
    attribute: ColumnPath
    value: Any


@dataclass(frozen=True)
class EqualNullSafe(Filter):
    attribute: ColumnPath
    value: Any


@dataclass(frozen=True)
class GreaterThan(Filter):
    attribute: ColumnPath
    value: Any


@dataclass(frozen=True)
class GreaterThanOrEqual(Filter):
    attribute: ColumnPath
    value: Any


@dataclass(frozen=True)
class LessThan(Filter):
    attribute: ColumnPath
    value: Any


@dataclass(frozen=True)
class LessThanOrEqual(Filter):
    attribute: ColumnPath
    value: Any


@dataclass(frozen=True)
class In(Filter):
    attribute: ColumnPath
    value: tuple[Any, ...]


@dataclass(frozen=True)
class IsNull(Filter):
    attribute: ColumnPath


@dataclass(frozen=True)
class IsNotNull(Filter):
    attribute: ColumnPath


@dataclass(frozen=True)
class Not(Filter):
    child: Filter


@dataclass(frozen=True)
class StringStartsWith(Filter):
    attribute: ColumnPath
    value: str


@dataclass(frozen=True)
class StringEndsWith(Filter):
    attribute: ColumnPath
    value: str


@dataclass(frozen=True)
class StringContains(Filter):
    attribute: ColumnPath
    value: str


# ---- the API classes ------------------------------------------------------------------------------


class InputPartition:
    """A partition of a read: ``value`` is anything picklable that ``read`` needs to read it."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        attributes = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"


class WriterCommitMessage:
    """What one write task returns to ``commit`` or ``abort``."""


class DataSource(ABC):  # noqa: B024  (Spark's methods all have defaults)
    """The entry point: a name, a schema, a reader and optionally a writer."""

    def __init__(self, options: dict[str, str]) -> None:
        self.options = options

    @classmethod
    def name(cls) -> str:
        """The short name of the source, as in ``spark.read.format(name)`` (default: the class name)."""
        return cls.__name__

    def schema(self) -> StructType | str:
        """The source's schema: a :class:`StructType` or a DDL string."""
        raise PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "schema"})

    def reader(self, schema: StructType) -> DataSourceReader:
        raise PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "reader"})

    def writer(self, schema: StructType, overwrite: bool) -> DataSourceWriter:
        raise PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "writer"})

    def streamReader(self, schema: StructType) -> Any:
        raise PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "streamReader"})

    def streamWriter(self, schema: StructType, overwrite: bool) -> Any:
        raise PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "streamWriter"})


class DataSourceReader(ABC):
    """Reads one source: ``partitions`` cuts it, ``read`` reads a partition."""

    def pushFilters(self, filters: list[Filter]) -> Iterable[Filter]:
        """Takes the filters it can handle and returns the ones it cannot (the default takes none)."""
        return filters

    def partitions(self) -> Sequence[InputPartition]:
        """The partitions of a read (the default is one ``InputPartition(None)``)."""
        return [InputPartition(None)]

    @abstractmethod
    def read(self, partition: InputPartition) -> Iterator[tuple[Any, ...]] | Iterator[Any]:
        """The rows of ``partition``: an iterator of tuples, ``Row`` objects or ``pyarrow.RecordBatch``."""
        ...


class DataSourceWriter(ABC):
    """Writes rows: ``write`` per task, then ``commit`` once, or ``abort`` once."""

    @abstractmethod
    def write(self, iterator: Iterator[Any]) -> WriterCommitMessage:
        """Writes the ``Row`` objects of one task and returns its commit message."""
        ...

    def commit(self, messages: list[WriterCommitMessage | None]) -> None:  # noqa: B027
        """Called once when every task succeeded."""

    def abort(self, messages: list[WriterCommitMessage | None]) -> None:  # noqa: B027
        """Called once when a task failed; a failed task's message is ``None``."""


class DataSourceArrowWriter(DataSourceWriter):
    """A writer that takes ``pyarrow.RecordBatch`` objects instead of rows."""


# ---- running them ---------------------------------------------------------------------------------


def _described(error: BaseException) -> str:
    """``error`` as the worker's ``handle_worker_exception`` writes it: the formatted traceback, its last line
    ``Type: message`` (``[CLASS] message`` for an errors-package error)."""
    import traceback

    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def _source_error(action: str, kind: str, error: BaseException | str) -> AnalysisException:
    """``PYTHON_DATA_SOURCE_ERROR`` (SQLSTATE 38000), an ``AnalysisException`` as Spark's
    ``QueryCompilationErrors`` raises it: "Failed to <action> Python data source <type>: <msg>"."""
    message = error if isinstance(error, str) else _described(error)
    return AnalysisException(
        errorClass="PYTHON_DATA_SOURCE_ERROR",
        messageParameters={"action": action, "type": kind, "msg": message},
        sqlState="38000",
    )


def _mismatch(error_class: str, **parameters: str) -> PySparkRuntimeError:
    """A PySpark-side class of the data source worker, its text from the class's template."""
    return PySparkRuntimeError(errorClass=error_class, messageParameters=parameters)


def _guard(action: str, kind: str, call: Any) -> Any:
    """``call()`` with every exception of the user's code (and the worker's own checks) raised once as
    ``PYTHON_DATA_SOURCE_ERROR`` of Spark's ``action`` and ``kind`` (``UserDefinedPythonDataSource.scala``)."""
    try:
        return call()
    except Exception as error:
        raise _source_error(action, kind, error) from error


def _struct_of(schema: StructType | str) -> StructType:
    if isinstance(schema, StructType):
        return schema
    if isinstance(schema, str):
        parsed = DataType.fromDDL(schema)
        if isinstance(parsed, StructType):
            return parsed
        return StructType().add("value", parsed)
    raise PySparkTypeError(
        errorClass="NOT_EXPECTED_TYPE",
        messageParameters={
            "arg_name": "schema",
            "expected_type": "StructType or str",
            "arg_type": type(schema).__name__,
        },
    )


class _Registered:
    """A registered class and the version the registration has (a re-registration bumps it)."""

    __slots__ = ("cls", "name", "version")

    def __init__(self, cls: type[DataSource], name: str, version: int) -> None:
        self.cls = cls
        self.name = name
        self.version = version


class DataSourceRegistration:
    """``spark.dataSource``: registers Python data source classes for the session."""

    def __init__(self, session: SparkSession) -> None:
        self._session = session

    def register(self, dataSource: type[DataSource]) -> None:
        """Registers ``dataSource`` under its ``name()``; a later registration of the name replaces it."""
        if not (isinstance(dataSource, type) and issubclass(dataSource, DataSource)):
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "dataSource",
                    "expected_type": "a subclass of DataSource",
                    "arg_type": type(dataSource).__name__,
                },
            )
        name = _guard("create", "instance", lambda: dataSource.name())
        if not isinstance(name, str) or not name:
            raise _mismatch(
                "DATA_SOURCE_TYPE_MISMATCH", expected="name() to return a non-empty string", actual=repr(name)
            )
        registry = _registry(self._session)
        old = registry.get(name.lower())
        registry[name.lower()] = _Registered(dataSource, name, (old.version + 1) if old else 1)


def _registry(session: SparkSession) -> dict[str, _Registered]:
    registry = getattr(session, "_python_sources", None)
    if registry is None:
        registry = {}
        session._python_sources = registry  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return registry


# The short names of Spark 4.3's `DataSourceRegister` services (sql/core, avro, kafka, hive, mllib) and the
# `org.apache.spark.` class names `lookupDataSource` resolves first. Checked by hand against the module list:
# re-diff the service files when the supported Spark version moves.
_BUILTIN_FORMATS = frozenset(
    {
        "parquet",
        "orc",
        "avro",
        "json",
        "csv",
        "text",
        "xml",
        "jdbc",
        "binaryfile",
        "image",
        "libsvm",
        "kafka",
        "hive",
        "noop",
        "console",
        "rate",
        "rate-micro-batch",
        "socket",
        "statestore",
        "state-metadata",
    }
)

# `DataSource.backwardCompatibilityMap` (v4.3.0): the class names Spark maps to a built-in source, lower-cased.
_BACKWARD_COMPATIBLE = frozenset(
    {
        "org.apache.spark.sql.jdbc",
        "org.apache.spark.sql.jdbc.defaultsource",
        "org.apache.spark.sql.execution.datasources.jdbc",
        "org.apache.spark.sql.execution.datasources.jdbc.defaultsource",
        "org.apache.spark.sql.execution.datasources.json",
        "org.apache.spark.sql.execution.datasources.json.defaultsource",
        "org.apache.spark.sql.execution.datasources.parquet",
        "org.apache.spark.sql.execution.datasources.parquet.defaultsource",
        "org.apache.spark.sql.parquet",
        "org.apache.spark.sql.hive.orc.defaultsource",
        "org.apache.spark.sql.hive.orc",
        "org.apache.spark.sql.execution.datasources.orc.defaultsource",
        "org.apache.spark.sql.execution.datasources.orc",
        "org.apache.spark.ml.source.libsvm.defaultsource",
        "org.apache.spark.ml.source.libsvm",
        "org.apache.spark.sql.execution.streaming.textsocketsourceprovider",
        "org.apache.spark.sql.execution.streaming.ratesourceprovider",
        "com.databricks.spark.csv",
        "com.databricks.spark.xml",
        "com.databricks.spark.avro",
    }
)


def registered(session: SparkSession, name: str) -> _Registered | None:
    """The class registered under ``name`` (any letter case), if any. Spark's built-in file sources win over a
    Python source of the same name (``DataSource.lookupDataSource``)."""
    lowered = name.lower()
    if lowered in _BUILTIN_FORMATS or lowered in _BACKWARD_COMPATIBLE:
        return None
    return _registry(session).get(name.lower())


def _read_partitions(reader: DataSourceReader) -> list[InputPartition]:
    """``reader.partitions()`` as Spark's planner takes it: one ``InputPartition(None)`` for a reader that
    does not implement it or returns an empty list; anything but a list of ``InputPartition`` is refused."""
    try:
        partitions = reader.partitions()
    except NotImplementedError:
        return [InputPartition(None)]
    if not isinstance(partitions, list):
        raise _mismatch(
            "DATA_SOURCE_TYPE_MISMATCH",
            expected="'partitions' to return a list",
            actual=f"'{type(partitions).__name__}'",
        )
    if not all(isinstance(p, InputPartition) for p in partitions):
        raise _mismatch(
            "DATA_SOURCE_TYPE_MISMATCH",
            expected="elements in 'partitions' to be of type 'InputPartition'",
            actual=", ".join(f"'{type(p).__name__}'" for p in partitions),
        )
    return partitions or [InputPartition(None)]


def _batches_of(pa: Any, output: Any, schema: StructType, session: SparkSession, source: DataSource) -> Iterator[Any]:
    """``reader.read``'s output as Arrow batches of the declared schema (Spark's ``records_to_arrow_batches``:
    the first element decides between the Arrow path, which only checks the schema, and the row path)."""
    from itertools import chain, islice

    from krishiv.sql.row import Row
    from krishiv.sql.session import _cell, _pa_schema, _pa_type, _verify_row

    arrow_schema = _pa_schema(schema, pa)
    names = schema.fieldNames()
    width = len(names)
    position = {name: i for i, name in enumerate(names)}

    def flush(rows: list[tuple[Any, ...]]) -> Any:
        columns = []
        for i, field in enumerate(schema.fields):
            values = [_cell(r[i], field.dataType, session) for r in rows]
            columns.append(pa.array(values, type=_pa_type(field.dataType, pa)))
        if not columns:
            # No array carries a length, so a batch without columns would lose its rows.
            return pa.RecordBatch.from_struct_array(pa.array([{} for _ in rows], type=pa.struct([])))
        return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)

    try:
        first = next(output)
    except StopIteration:
        return
    if isinstance(first, pa.RecordBatch):
        if first.num_columns != width:
            raise _mismatch("DATA_SOURCE_RETURN_SCHEMA_MISMATCH", expected=str(width), actual=str(first.num_columns))
        for name in names:
            if name not in first.schema.names:
                raise _mismatch(
                    "DATA_SOURCE_RETURN_SCHEMA_MISMATCH", expected=str(names), actual=str(first.schema.names)
                )
        if not arrow_schema.equals(first.schema):
            raise _mismatch("DATA_SOURCE_RETURN_SCHEMA_MISMATCH", expected=str(arrow_schema), actual=str(first.schema))
        yield first
        yield from output
        return

    iterator = chain([first], output)
    while batch := list(islice(iterator, MAX_BATCH_ROWS)):
        rows: list[tuple[Any, ...]] = []
        for item in batch:
            if hasattr(item, "__len__") and len(item) != width:
                raise _mismatch("DATA_SOURCE_RETURN_SCHEMA_MISMATCH", expected=str(width), actual=str(len(item)))
            if not isinstance(item, list | tuple):
                raise _mismatch(
                    "DATA_SOURCE_INVALID_RETURN_TYPE",
                    type=type(item).__name__,
                    name=source.name(),
                    supported_types="tuple, list, `pyspark.sql.types.Row`, `pyarrow.RecordBatch`",
                )
            if isinstance(item, Row) and hasattr(item, "__fields__"):
                # A named Row is placed by field name, not by position.
                if set(item.__fields__) != set(names):
                    raise _mismatch(
                        "DATA_SOURCE_RETURN_SCHEMA_MISMATCH", expected=str(names), actual=str(item.__fields__)
                    )
                values: list[Any] = [None] * width
                for name in names:
                    values[position[name]] = item[name]
            else:
                values = list(item)
            _verify_row(schema, values)
            rows.append(tuple(values))
        yield flush(rows)


def _plan_and_read(
    session: SparkSession,
    source: DataSource,
    schema: StructType,
    label: str,
    filters: list[Filter] | None = None,
    pushdown: bool | None = None,
) -> tuple[list[bool], Any]:
    """One scan's planning and read: ``reader()``, ``pushFilters(filters)`` when there are filters,
    ``partitions()``; returns ``(pushed, read)``: per filter whether the reader took it (so the engine does
    not apply it again) and the callable that reads each partition as one Arrow table.

    Planning failures (``reader()``, its type, ``pushFilters``, its answer, ``partitions()``) are raised once as
    ``PYTHON_DATA_SOURCE_ERROR`` of Spark's pairs; what ``read`` raises is not wrapped here (the engine
    raises it as ``PYTHON_EXCEPTION``, as a failed read task)."""
    from krishiv.sql import _convert
    from krishiv.sql.session import _pa_schema

    pa = _convert.require_pyarrow("a Python data source read")
    offered = filters or []
    pushed = [False] * len(offered)
    # One planning call uses one setting: the caller's reading when it made one.
    pushdown_on = _pushdown_enabled(session) if pushdown is None else pushdown

    def initialize() -> DataSourceReader:
        reader = source.reader(schema)
        if not isinstance(reader, DataSourceReader):
            raise PySparkAssertionError(
                errorClass="DATA_SOURCE_TYPE_MISMATCH",
                messageParameters={
                    "expected": "an instance of DataSourceReader",
                    "actual": f"'{type(reader).__name__}'",
                },
            )
        if getattr(reader.pushFilters, "__func__", None) is not DataSourceReader.pushFilters:
            if not pushdown_on:
                raise PySparkAssertionError(
                    errorClass="DATA_SOURCE_PUSHDOWN_DISABLED",
                    messageParameters={"type": type(reader).__name__, "conf": PUSHDOWN_CONF},
                )
            if offered:
                rejected = reader.pushFilters(list(offered))
                kept = list(rejected) if rejected is not None else []
                if any(not any(r is f for f in offered) for r in kept):
                    raise PySparkAssertionError(
                        errorClass="DATA_SOURCE_EXTRANEOUS_FILTERS",
                        messageParameters={"type": type(reader).__name__},
                    )
                for i, f in enumerate(offered):
                    pushed[i] = not any(r is f for r in kept)
        return reader

    reader = _guard("initialize", "reader", initialize)
    partitions = _guard("generate", "read partitions", lambda: _read_partitions(reader))

    def read() -> Any:
        batches: list[Any] = []
        held = 0
        for partition in partitions:
            output = reader.read(partition)
            if not isinstance(output, Iterator):
                raise _mismatch(
                    "DATA_SOURCE_INVALID_RETURN_TYPE",
                    type=type(output).__name__,
                    name=source.name(),
                    supported_types="iterator",
                )
            for batch in _batches_of(pa, output, schema, session, source):
                held += batch.nbytes
                if held > MAX_SOURCE_BYTES:  # before the next batch is pulled: the source is never read whole
                    raise PySparkRuntimeError(
                        errorClass="KRISHIV_PYTHON_DATA_SOURCE_LIMIT",
                        messageParameters={"source": label, "bytes": str(held), "limit": str(MAX_SOURCE_BYTES)},
                        sqlState="53400",
                    )
                batches.append(batch)
        return pa.Table.from_batches(batches, schema=_pa_schema(schema, pa))

    return pushed, read


_FILTERS: dict[str, type[Filter]] = {
    cls.__name__: cls
    for cls in (
        EqualTo,
        EqualNullSafe,
        GreaterThan,
        GreaterThanOrEqual,
        LessThan,
        LessThanOrEqual,
        In,
        IsNull,
        IsNotNull,
        StringStartsWith,
        StringEndsWith,
        StringContains,
    )
}


def _filter_of(spec: tuple[Any, ...]) -> Filter:
    """A filter object from the engine's ``(kind, attribute path, value)``."""
    kind, attribute, value = spec
    if kind == "Not":
        return Not(_filter_of(value))
    cls: Any = _FILTERS[kind]
    if kind in ("IsNull", "IsNotNull"):
        return cls(attribute)
    return cls(attribute, value)


def _pushdown_enabled(session: SparkSession) -> bool:
    try:
        return str(session.conf.get(PUSHDOWN_CONF, "false")).lower() == "true"
    except Exception:
        return False


def load(session: SparkSession, entry: _Registered, options: dict[str, str], user_schema: Any) -> DataFrame:
    """``spark.read.format(name).load()`` over a registered Python source."""
    from krishiv.sql import _convert
    from krishiv.sql.dataframe import DataFrame
    from krishiv.sql.session import _pa_schema

    pa = _convert.require_pyarrow("a Python data source read")
    source = _guard("create", "instance", lambda: entry.cls(options=CaseInsensitiveDict(options)))
    if user_schema is not None:
        schema = _struct_of(user_schema)
    else:
        schema = _struct_of(_guard("create", "instance", source.schema))
    empty = pa.Table.from_batches([], schema=_pa_schema(schema, pa))
    # The callback lives in the native catalog, which the session owns: hold the session weakly.
    ref = weakref.ref(session)

    def owner_of() -> SparkSession:
        owner = ref()
        if owner is None:
            raise PySparkRuntimeError(errorClass="NO_ACTIVE_SESSION", messageParameters={})
        return owner

    def read() -> Any:
        return _plan_and_read(owner_of(), source, schema, entry.name)[1]()

    def plan(specs: list[tuple[Any, ...]]) -> tuple[list[bool], Any]:
        owner = owner_of()
        enabled = _pushdown_enabled(owner)
        if not enabled:
            # Nothing is offered (Spark raises DATA_SOURCE_PUSHDOWN_DISABLED at the action, not at explain()).
            return [False] * len(specs), read
        return _plan_and_read(owner, source, schema, entry.name, [_filter_of(spec) for spec in specs], enabled)

    native = session._native.register_python_source(entry.name, empty, read, plan)
    return DataFrame(native, session)


def save(session: SparkSession, entry: _Registered, df: DataFrame, mode: str, options: dict[str, str]) -> None:
    """``df.write.format(name).mode(mode).save()`` through a registered Python source's writer."""
    import traceback

    from krishiv.sql import _convert

    lowered = mode.lower()
    if lowered in ("error", "errorifexists", "default", "ignore"):
        # Spark's `AnalysisException`, its `createMode` the `SaveMode` name.
        raise AnalysisException(
            errorClass="UNSUPPORTED_DATA_SOURCE_SAVE_MODE",
            messageParameters={
                "source": entry.name,
                "createMode": "Ignore" if lowered == "ignore" else "ErrorIfExists",
            },
            sqlState="0A000",
        )
    if lowered not in ("append", "overwrite"):
        raise PySparkRuntimeError(
            f'[INVALID_SAVE_MODE] The specified save mode "{mode}" is invalid.',
            errorClass="INVALID_SAVE_MODE",
            messageParameters={"mode": mode},
        )
    overwrite = lowered == "overwrite"
    source = _guard("create", "instance", lambda: entry.cls(options=CaseInsensitiveDict(options)))
    struct = df.schema
    writer = _guard("initialize", "writer", lambda: _writer_of(source, struct, overwrite))
    arrow_writer = isinstance(writer, DataSourceArrowWriter)
    message: WriterCommitMessage | None = None
    failure: BaseException | None = None
    user_failure = False
    try:
        table = df.toArrow()
        data = iter(table.to_batches()) if arrow_writer else iter(_convert.rows_of(table))
        try:
            message = writer.write(data)
        except Exception:
            user_failure = True
            raise
    except BaseException as error:  # the task failed: abort, then raise
        failure = error
        traced = traceback.format_exc()
    if failure is not None:
        _guard("commit or abort", "write", lambda: writer.abort([None]))
        if not user_failure:
            raise failure  # an engine error or an interrupt reaches the caller unchanged
        raise SparkRuntimeException(
            errorClass="PYTHON_EXCEPTION",
            messageParameters={"msg": "An exception was thrown from the Python worker", "traceback": traced},
            sqlState="38000",
        ) from failure
    _guard("commit or abort", "write", lambda: writer.commit([message]))


def _writer_of(source: DataSource, schema: StructType, overwrite: bool) -> DataSourceWriter:
    writer = source.writer(schema, overwrite)
    if not isinstance(writer, DataSourceWriter):
        raise PySparkAssertionError(
            errorClass="DATA_SOURCE_TYPE_MISMATCH",
            messageParameters={"expected": "an instance of DataSourceWriter", "actual": f"'{type(writer).__name__}'"},
        )
    return writer
