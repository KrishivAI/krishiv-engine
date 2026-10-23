# SPDX-License-Identifier: Apache-2.0
"""``SparkSession``: the entry point, as ``pyspark.sql.SparkSession`` (embedded: no server)."""

from __future__ import annotations

import datetime
import threading
import warnings
from decimal import Decimal
from typing import Any

from krishiv import _native
from krishiv.errors import not_expected_type, not_implemented, unsupported_feature
from krishiv.sql.conf import RuntimeConfig
from krishiv.sql.dataframe import DataFrame
from krishiv.sql.readwriter import DataFrameReader
from krishiv.sql.types import DataType, StructType

#: The Spark version this engine is compatible with.
SPARK_VERSION = "4.2.0"

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

    builder: Builder = _BuilderProperty()  # ty: ignore[invalid-assignment]

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

    @property
    def catalog(self) -> Catalog:
        """The catalog of temporary views, as ``pyspark.sql.catalog.Catalog``.

        The embedded engine has no databases: every view is temporary and
        listed under the ``default`` database.
        """
        return Catalog(self)

    def sql(self, sqlQuery: str, args: Any = None, **kwargs: Any) -> DataFrame:
        """Parses and analyzes ``sqlQuery`` now."""
        if not isinstance(sqlQuery, str):
            raise not_expected_type("sqlQuery", "str", sqlQuery)
        if args is not None or kwargs:
            raise not_implemented("sql with parameters")
        return DataFrame(self._native.sql(sqlQuery), self)

    def table(self, tableName: str) -> DataFrame:
        return DataFrame(self._native.table(tableName), self)

    def range(
        self,
        start: int,
        end: int | None = None,
        step: int = 1,
        numPartitions: int | None = None,
    ) -> DataFrame:
        """A :class:`DataFrame` with a single ``bigint`` column ``id``, holding
        ``start`` (inclusive) to ``end`` (exclusive) by ``step``.

        With one argument it is the end (from 0); ``numPartitions`` is
        accepted and ignored (one node has nothing to partition).

        Examples
        --------
        >>> spark.range(1, 7, 2).show()
        +---+
        | id|
        +---+
        |  1|
        |  3|
        |  5|
        +---+

        >>> spark.range(3).show()
        +---+
        | id|
        +---+
        |  0|
        |  1|
        |  2|
        +---+
        """
        if end is None:
            start, end = 0, start
        for name, value in (("start", start), ("end", end), ("step", step)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise not_expected_type(name, "int", value)
        if step == 0:
            raise ValueError("step cannot be 0")
        if numPartitions is not None and (not isinstance(numPartitions, int) or isinstance(numPartitions, bool)):
            raise not_expected_type("numPartitions", "int", numPartitions)
        from krishiv.sql import _convert

        pa = _convert.require_pyarrow("range")
        table = pa.table({"id": pa.array(list(range(start, end, step)), type=pa.int64())})
        return DataFrame(self._native.create_dataframe(table), self)

    def createDataFrame(
        self,
        data: Any,
        schema: Any = None,
        samplingRatio: float | None = None,
        verifySchema: bool = True,
    ) -> DataFrame:
        """A DataFrame of ``data``: an object with ``__arrow_c_stream__`` or
        ``__arrow_c_array__`` (a pyarrow Table, a Polars or DuckDB result, ...),
        a pandas DataFrame, or rows (tuples, lists, dicts, :class:`Row`) with
        an explicit ``schema``.

        ``schema`` is a :class:`StructType`, a DDL string (``"a int, b string"``),
        a list of column names (types are inferred from the rows), or a single
        atomic type (values become one ``value`` column). With ``verifySchema``
        (the default), every row is checked against the schema first;
        ``samplingRatio`` is accepted and ignored (it only affects RDD input,
        which the embedded session does not take).

        Examples
        --------
        >>> spark.createDataFrame([("Alice", 1)], ["name", "age"]).show()
        +-----+---+
        | name|age|
        +-----+---+
        |Alice|  1|
        +-----+---+

        >>> schema = StructType([StructField("name", StringType(), True), StructField("age", IntegerType(), True)])
        >>> spark.createDataFrame([("Alice", 1)], schema).show()
        +-----+---+
        | name|age|
        +-----+---+
        |Alice|  1|
        +-----+---+

        >>> spark.createDataFrame([("Alice", 1)], "name: string, age: int").show()
        +-----+---+
        | name|age|
        +-----+---+
        |Alice|  1|
        +-----+---+
        """
        from krishiv.sql.row import Row

        if isinstance(data, DataFrame):
            raise TypeError("createDataFrame does not take a DataFrame")
        if _is_pandas(data):
            # pandas also speaks the Arrow capsule protocol: check it first so
            # an explicit schema applies, as in PySpark.
            return self._from_pandas(data, schema)
        if hasattr(data, "__arrow_c_stream__") or hasattr(data, "__arrow_c_array__"):
            if schema is not None:
                raise not_implemented("createDataFrame with a schema (for Arrow-producing objects)")
            return DataFrame(self._native.create_dataframe(data), self)
        rows = list(data) if not isinstance(data, list) else data
        if schema is None:
            raise not_implemented("createDataFrame without an explicit schema (schema inference)")
        return self._from_rows(rows, schema, verifySchema, Row)

    def _from_pandas(self, data: Any, schema: Any) -> DataFrame:
        from krishiv.sql import _convert

        pa = _convert.require_pyarrow("createDataFrame")
        table = pa.Table.from_pandas(data)
        if schema is None:
            return DataFrame(self._native.create_dataframe(table), self)
        if isinstance(schema, list | tuple):
            if len(schema) != table.num_columns or not all(isinstance(n, str) for n in schema):
                raise not_expected_type("schema", "list of str", schema)
            return DataFrame(self._native.create_dataframe(table.rename_columns(list(schema))), self)
        target = _pa_schema(self._explicit_schema(schema), pa)
        return DataFrame(self._native.create_dataframe(table.cast(target)), self)

    def _from_rows(self, rows: list[Any], schema: Any, verify: bool, Row: Any) -> DataFrame:
        from krishiv.sql import _convert

        pa = _convert.require_pyarrow("createDataFrame")
        if isinstance(schema, list | tuple):
            return self._from_rows_inferred(rows, schema, pa, Row)
        struct = self._explicit_schema(schema)
        named = [f.name for f in struct.fields]
        width = len(named)
        valued = [_row_values(row, named, width, Row) for row in rows]
        if verify:
            for values in valued:
                _verify_row(struct, values)
        columns = []
        for i, field in enumerate(struct.fields):
            arrow_type = _pa_type(field.dataType, pa)
            values = [_cell(v[i], field.dataType, self) for v in valued]
            columns.append(pa.array(values, type=arrow_type))
        table = pa.table(columns, names=named)
        return DataFrame(self._native.create_dataframe(table), self)

    def _from_rows_inferred(self, rows: list[Any], names: list[Any] | tuple[Any, ...], pa: Any, Row: Any) -> DataFrame:
        """Rows with a list of column names: types come from the rows, as
        pyarrow infers them (ints are ``bigint``, as in PySpark)."""

        for n in names:
            if not isinstance(n, str):
                raise not_expected_type("schema", "list of str", n)
        if not names or not rows:
            from krishiv.errors import PySparkValueError

            raise PySparkValueError(errorClass="CANNOT_INFER_EMPTY_SCHEMA")
        valued = [_row_values(row, list(names), len(names), Row) for row in rows]
        arrays = [pa.array([v[i] for v in valued]) for i in range(len(names))]
        return DataFrame(self._native.create_dataframe(pa.table(arrays, names=list(names))), self)

    def _explicit_schema(self, schema: Any) -> StructType:
        if isinstance(schema, StructType):
            return schema
        if isinstance(schema, str):
            parsed = DataType.fromDDL(schema)
            if isinstance(parsed, StructType):
                return parsed
            return StructType().add("value", parsed)
        if isinstance(schema, DataType):
            return StructType().add("value", schema)
        raise not_expected_type("schema", "StructType, str, list of str or DataType", schema)

    def emptyDataFrame(self, schema: StructType | str) -> DataFrame:
        """An empty :class:`DataFrame` with ``schema`` (a :class:`StructType`
        or a DDL string), as ``pyspark.sql.SparkSession.emptyDataFrame``.

        Examples
        --------
        >>> df = spark.emptyDataFrame("name STRING, age INT")
        >>> df.printSchema()
        root
         |-- name: string (nullable = true)
         |-- age: int (nullable = true)
        >>> df.count()
        0
        """
        if not isinstance(schema, StructType | str):
            raise not_expected_type("schema", "StructType or str", schema)
        return self.createDataFrame([], schema)

    def newSession(self) -> SparkSession:
        """A new session with separate SQL configuration and temporary views,
        as ``pyspark.sql.SparkSession.newSession``."""
        return SparkSession.builder.create()

    def stop(self) -> None:
        """Ends this session; the next ``getOrCreate`` makes a new one."""
        global _active
        with _lock:
            if _active is self:
                _active = None


def _is_pandas(data: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.DataFrame)


def _pa_type(data_type: DataType, pa: Any) -> Any:
    """The Arrow type holding ``data_type`` values when ``createDataFrame`` converts rows."""
    from krishiv.sql.types import (
        BinaryType,
        BooleanType,
        ByteType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        NullType,
        ShortType,
        StringType,
        TimestampNTZType,
        TimestampType,
    )

    if isinstance(data_type, NullType):
        return pa.null()
    if isinstance(data_type, BooleanType):
        return pa.bool_()
    if isinstance(data_type, ByteType):
        return pa.int8()
    if isinstance(data_type, ShortType):
        return pa.int16()
    if isinstance(data_type, IntegerType):
        return pa.int32()
    if isinstance(data_type, LongType):
        return pa.int64()
    if isinstance(data_type, FloatType):
        return pa.float32()
    if isinstance(data_type, DoubleType):
        return pa.float64()
    if isinstance(data_type, StringType):
        return pa.string()
    if isinstance(data_type, BinaryType):
        return pa.binary()
    if isinstance(data_type, DecimalType):
        return pa.decimal128(data_type.precision, data_type.scale)
    if isinstance(data_type, DateType):
        return pa.date32()
    if isinstance(data_type, TimestampType):
        return pa.timestamp("us", tz="UTC")
    if isinstance(data_type, TimestampNTZType):
        return pa.timestamp("us")
    raise unsupported_feature(f"createDataFrame with {data_type.simpleString()} columns")


def _pa_schema(struct: StructType, pa: Any) -> Any:
    import pyarrow as _pa

    return _pa.schema([_pa.field(f.name, _pa_type(f.dataType, pa), nullable=f.nullable) for f in struct.fields])


def _row_values(row: Any, names: list[str] | None, width: int, Row: Any) -> list[Any]:
    """One row of ``createDataFrame`` data as ``width`` values: a tuple, list
    or positional ``Row`` in order; a dict or named ``Row`` by field name
    (missing keys are NULL); any other value is a single-column row."""
    if isinstance(row, Row):
        if hasattr(row, "__fields__"):
            by_name = row.asDict()
            values = [by_name.get(n) for n in names] if names is not None else list(by_name.values())
        else:
            values = list(row)
    elif isinstance(row, dict):
        values = list(row.values()) if names is None else [row.get(n) for n in names]
    elif isinstance(row, list | tuple):
        values = list(row)
    else:
        values = [row]
    if len(values) != width:
        raise ValueError(f"Row has {len(values)} values, schema has {width} fields")
    return values


def _cell(value: Any, data_type: DataType, session: SparkSession) -> Any:
    """One ``createDataFrame`` value as pyarrow takes it (naive datetimes in
    the session time zone, as Spark reads them)."""
    from krishiv.sql import _convert
    from krishiv.sql.types import TimestampNTZType, TimestampType

    if isinstance(value, datetime.datetime):
        if isinstance(data_type, TimestampType):
            if value.tzinfo is None:
                return value.replace(tzinfo=_convert.session_zone(_session_zone(session)))
            return value
        if isinstance(data_type, TimestampNTZType):
            zone = _convert.session_zone(_session_zone(session))
            return value.astimezone(zone).replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value
    return value


def _session_zone(session: SparkSession) -> str:
    return session.conf.get("spark.sql.session.timeZone", "UTC") or "UTC"


def _verify_row(struct: StructType, values: list[Any]) -> None:
    """Checks one row against ``struct``, as ``verifySchema`` does."""
    from krishiv.errors import PySparkTypeError
    from krishiv.sql.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        FractionalType,
        IntegralType,
        StringType,
        TimestampNTZType,
        TimestampType,
    )

    for field, value in zip(struct.fields, values, strict=True):
        if value is None:
            if not field.nullable:
                raise PySparkTypeError(f"field {field.name}: This field is not nullable, but got None")
            continue
        ok = False
        t = field.dataType
        if isinstance(t, BooleanType):
            ok = isinstance(value, bool)
        elif isinstance(t, IntegralType):
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(t, DecimalType):
            ok = isinstance(value, Decimal | int) and not isinstance(value, bool)
        elif isinstance(t, FractionalType):
            ok = isinstance(value, int | float) and not isinstance(value, bool)
        elif isinstance(t, StringType):
            ok = isinstance(value, str)
        elif isinstance(t, BinaryType):
            ok = isinstance(value, bytes | bytearray)
        elif isinstance(t, DateType):
            ok = isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)
        elif isinstance(t, TimestampType | TimestampNTZType):
            ok = isinstance(value, datetime.datetime)
        else:
            ok = True  # nested and other types: the Arrow conversion reports them
        if not ok:
            raise PySparkTypeError(
                f"field {field.name}: {t.simpleString()} can not accept object {value!r} in type {type(value).__name__}"
            )


class Catalog:
    """The catalog of temporary views, as ``pyspark.sql.catalog.Catalog``.

    The embedded engine has no databases or persistent tables: every table
    is a temporary view in the ``default`` database.
    """

    def __init__(self, session: SparkSession) -> None:
        self._session = session

    def currentDatabase(self) -> str:
        """The current database: always ``default`` (there is no other)."""
        return "default"

    def listDatabases(self) -> DataFrame:
        """The databases: just ``default``."""
        return self._session.createDataFrame([("default",)], ["name"])

    def tableExists(self, tableName: str, dbName: str | None = None) -> bool:
        """Whether the temporary view ``tableName`` exists."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        names = self._session._native.temp_table_names()
        return any(n.lower() == tableName.lower() for n in names)

    def listTables(self, dbName: str | None = None) -> DataFrame:
        """The temporary views as a DataFrame of ``(name, database,
        description, tableType, isTemporary)`` rows.

        .. note:: The engine has no catalogs or namespaces, so this is the
            classic five-column shape rather than Spark 4.x's namespaced one.
        """
        if dbName is not None and not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        rows = [(n, "default", None, "TEMPORARY", True) for n in self._session._native.temp_table_names()]
        return self._session.createDataFrame(
            rows,
            "name string, database string, description string, tableType string, isTemporary boolean",
        )

    def dropTempView(self, viewName: str) -> bool:
        """Drops the temporary view ``viewName``; whether it existed."""
        if not isinstance(viewName, str):
            raise not_expected_type("viewName", "str", viewName)
        return self._session._native.drop_temp_view(viewName)
