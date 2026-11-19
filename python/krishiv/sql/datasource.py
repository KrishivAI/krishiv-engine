# SPDX-License-Identifier: Apache-2.0
"""The Python Data Source API (batch), shaped like ``pyspark.sql.datasource`` of Spark 4.3.0.

A user subclasses :class:`DataSource` (and :class:`DataSourceReader`, :class:`DataSourceWriter` or
:class:`DataSourceArrowWriter`), registers the class with ``spark.dataSource.register(cls)`` and then
reads with ``spark.read.format(name)`` and writes with ``df.write.format(name)``.

How it runs here (design P3-T040):

* The schema is fixed when ``load`` is called (Spark's ``create_data_source`` step).
* The reader's ``partitions()`` and ``read()`` run when the query is *planned*, once per physical plan, so
  a source that lists files sees new ones on the next action (Spark plans the read per execution).
  Partitions run one after the other in this process (the process pool's parallel partitions are T014).
* ``read`` may yield tuples, ``Row`` objects, lists or ``pyarrow.RecordBatch`` objects; they are cast to
  the declared schema.
* A write runs as one task over every row of the DataFrame, then ``commit`` (or ``abort`` when the task
  failed, with ``None`` for the failed task's message).
* Filter pushdown: ``spark.sql.python.filterPushdown.enabled`` is off by default, as in Spark 4.3.0; a
  reader that overrides ``pushFilters`` then fails with ``DATA_SOURCE_PUSHDOWN_DISABLED``. With the
  switch on the engine does not push filters into the reader yet (it applies every filter itself).
* The streaming classes (``DataSourceStreamReader``, ``SimpleDataSourceStreamReader``,
  ``DataSourceStreamWriter``, ``DataSourceStreamArrowWriter``) are P4: ``streamReader`` and
  ``streamWriter`` raise ``NOT_IMPLEMENTED``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from krishiv.errors import (
    PySparkNotImplementedError,
    PySparkRuntimeError,
    PySparkTypeError,
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

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key.lower(), default)

    def pop(self, key: str, *default: Any) -> Any:
        return super().pop(key.lower(), *default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return super().setdefault(key.lower(), default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
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
        return f"{self.__class__.__name__}({self.value!r})"


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
        """Takes the filters it can handle and returns the ones it cannot."""
        raise PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "pushFilters"})

    def partitions(self) -> Sequence[InputPartition]:
        raise PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "partitions"})

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


def _source_error(action: str, kind: str, error: BaseException) -> PySparkRuntimeError:
    """``PYTHON_DATA_SOURCE_ERROR`` (SQLSTATE 38000): "Failed to <action> Python data source <type>: <msg>"."""
    message = f"{type(error).__name__}: {error}"
    return PySparkRuntimeError(
        f"[PYTHON_DATA_SOURCE_ERROR] Failed to {action} Python data source {kind}: {message}",
        errorClass="PYTHON_DATA_SOURCE_ERROR",
        messageParameters={"action": action, "type": kind, "msg": message},
        sqlState="38000",
    )


def _mismatch(feature: str, message: str) -> PySparkRuntimeError:
    return PySparkRuntimeError(
        f"[{feature}] {message}",
        errorClass=feature,
        messageParameters={"message": message},
    )


def _guard(action: str, kind: str, call: Any) -> Any:
    """``call()``, any exception of the user's code raised as ``PYTHON_DATA_SOURCE_ERROR``."""
    try:
        return call()
    except PySparkRuntimeError:
        raise
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
            raise _mismatch("DATA_SOURCE_TYPE_MISMATCH", "the data source name() must be a non-empty string")
        registry = _registry(self._session)
        old = registry.get(name.lower())
        registry[name.lower()] = _Registered(dataSource, name, (old.version + 1) if old else 1)


def _registry(session: SparkSession) -> dict[str, _Registered]:
    registry = getattr(session, "_python_sources", None)
    if registry is None:
        registry = {}
        session._python_sources = registry  # ty: ignore[unresolved-attribute]
    return registry


def registered(session: SparkSession, name: str) -> _Registered | None:
    """The class registered under ``name`` (any letter case), if any."""
    return _registry(session).get(name.lower())


def _read_partitions(reader: DataSourceReader) -> list[InputPartition]:
    """``reader.partitions()`` as Spark's planner takes it: one ``InputPartition(None)`` for a reader that
    does not implement it or returns an empty list; anything but a list of ``InputPartition`` is refused."""
    try:
        partitions = reader.partitions()
    except NotImplementedError:
        return [InputPartition(None)]
    except Exception as error:
        raise _source_error("generate", "read partitions", error) from error
    if not isinstance(partitions, list):
        raise _mismatch(
            "DATA_SOURCE_TYPE_MISMATCH",
            f"Expected the partitions() of the reader to be a list, but got {type(partitions).__name__}.",
        )
    for p in partitions:
        if not isinstance(p, InputPartition):
            raise _mismatch(
                "DATA_SOURCE_TYPE_MISMATCH",
                f"Expected every element of partitions() to be an InputPartition, but got {type(p).__name__}.",
            )
    return partitions or [InputPartition(None)]


def _batches_of(pa: Any, output: Any, schema: StructType, session: SparkSession) -> Iterator[Any]:
    """``reader.read``'s output as Arrow batches of the declared schema (Spark's ``records_to_arrow_batches``)."""
    from krishiv.sql.row import Row
    from krishiv.sql.session import _cell, _pa_schema, _pa_type, _verify_row

    arrow_schema = _pa_schema(schema, pa)
    width = len(schema.fields)

    def flush(rows: list[tuple[Any, ...]]) -> Any:
        columns = []
        for i, field in enumerate(schema.fields):
            values = [_cell(r[i], field.dataType, session) for r in rows]
            columns.append(pa.array(values, type=_pa_type(field.dataType, pa)))
        return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)

    rows: list[tuple[Any, ...]] = []
    for item in output:
        if isinstance(item, pa.RecordBatch):
            if rows:
                yield flush(rows)
                rows = []
            if item.num_columns != width:
                raise _mismatch(
                    "DATA_SOURCE_RETURN_SCHEMA_MISMATCH",
                    f"Return schema mismatch in the result from 'read' method. Expected: {width} columns, "
                    f"Found: {item.num_columns} columns. Make sure the returned values match the "
                    "required output schema.",
                )
            table = pa.Table.from_batches([item]).rename_columns(arrow_schema.names).cast(arrow_schema)
            yield from table.to_batches()
            continue
        if isinstance(item, Row | tuple | list):
            values = tuple(item)
        else:
            raise _mismatch(
                "DATA_SOURCE_INVALID_RETURN_TYPE",
                "Unsupported return type ('" + type(item).__name__ + "') from Python data source 'read'. "
                "Expected types: ('tuple', 'list', 'Row', 'pyarrow.RecordBatch').",
            )
        if len(values) != width:
            raise _mismatch(
                "DATA_SOURCE_RETURN_SCHEMA_MISMATCH",
                f"Return schema mismatch in the result from 'read' method. Expected: {width} columns, "
                f"Found: {len(values)} columns. Make sure the returned values match the required output schema.",
            )
        _verify_row(schema, list(values))
        rows.append(values)
        if len(rows) >= MAX_BATCH_ROWS:
            yield flush(rows)
            rows = []
    if rows:
        yield flush(rows)


def _plan_and_read(session: SparkSession, source: DataSource, schema: StructType) -> Any:
    """One physical plan's read: ``reader()``, ``partitions()``, ``read()`` of each, as one Arrow table."""
    from krishiv.sql import _convert
    from krishiv.sql.session import _pa_schema

    pa = _convert.require_pyarrow("a Python data source read")
    reader = _guard("create", "reader", lambda: source.reader(schema))
    if not isinstance(reader, DataSourceReader):
        raise _mismatch(
            "DATA_SOURCE_TYPE_MISMATCH",
            f"Expected an instance of DataSourceReader from the data source's reader(), got {type(reader).__name__}.",
        )
    if type(reader).pushFilters is not DataSourceReader.pushFilters and not _pushdown_enabled(session):
        raise _mismatch(
            "DATA_SOURCE_PUSHDOWN_DISABLED",
            f"Pushdown is disabled but the reader {type(reader).__name__} implements pushFilters. "
            f"Set {PUSHDOWN_CONF} to true to enable it.",
        )
    batches: list[Any] = []
    for partition in _read_partitions(reader):
        try:
            output = reader.read(partition)
        except Exception as error:
            raise _source_error("read", "partition", error) from error
        if not isinstance(output, Iterator):
            raise _mismatch(
                "DATA_SOURCE_INVALID_RETURN_TYPE",
                "Unsupported return type ('" + type(output).__name__ + "') from Python data source 'read'. "
                "Expected an iterator (a generator is fine); a list is not accepted.",
            )
        try:
            batches.extend(_batches_of(pa, output, schema, session))
        except PySparkRuntimeError:
            raise
        except Exception as error:
            raise _source_error("read", "partition", error) from error
    return pa.Table.from_batches(batches, schema=_pa_schema(schema, pa))


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
    native = session._native.register_python_source(empty, lambda: _plan_and_read(session, source, schema))
    return DataFrame(native, session)


def save(session: SparkSession, entry: _Registered, df: DataFrame, mode: str, options: dict[str, str]) -> None:
    """``df.write.format(name).mode(mode).save()`` through a registered Python source's writer."""
    from krishiv.sql.row import Row

    lowered = mode.lower()
    if lowered in ("error", "errorifexists", "default", "ignore"):
        raise PySparkRuntimeError(
            f'[UNSUPPORTED_DATA_SOURCE_SAVE_MODE] The data source "{entry.name}" cannot be written in the '
            f'{lowered} mode. Please use either the "Append" or "Overwrite" mode instead.',
            errorClass="UNSUPPORTED_DATA_SOURCE_SAVE_MODE",
            messageParameters={"source": entry.name, "createMode": f'"{lowered}"'},
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
    writer = _guard("initialize", "writer", lambda: source.writer(struct, overwrite))
    if not isinstance(writer, DataSourceWriter):
        raise _mismatch(
            "DATA_SOURCE_TYPE_MISMATCH",
            f"Expected an instance of DataSourceWriter from the data source's writer(), got {type(writer).__name__}.",
        )
    arrow_writer = isinstance(writer, DataSourceArrowWriter)
    message: WriterCommitMessage | None = None
    failure: BaseException | None = None
    try:
        table = df.toArrow()
        message = writer.write(iter(table.to_batches()) if arrow_writer else iter(_rows_of(table, Row)))
    except BaseException as error:  # the task failed: abort, then raise
        failure = error
    if failure is not None:
        try:
            writer.abort([None])
        except Exception as error:
            raise _source_error("commit or abort", "write", error) from error
        if isinstance(failure, PySparkRuntimeError):
            raise failure
        raise _source_error("write", "task", failure) from failure
    try:
        writer.commit([message])
    except Exception as error:
        raise _source_error("commit or abort", "write", error) from error


def _rows_of(table: Any, row_type: Any) -> Iterator[Any]:
    names = table.schema.names
    for record in table.to_pylist():
        yield row_type(**{n: record[n] for n in names}) if len(set(names)) == len(names) else row_type(*record.values())
