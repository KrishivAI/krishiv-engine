# SPDX-License-Identifier: Apache-2.0
"""``SparkSession``: the entry point, as ``pyspark.sql.SparkSession`` (embedded: no server)."""

from __future__ import annotations

import sys
import threading
import warnings
from typing import TYPE_CHECKING, Any

from krishiv import _native
from krishiv.errors import (
    AnalysisException,
    PySparkTypeError,
    PySparkValueError,
    not_expected_type,
    not_implemented,
    unsupported_feature,
)
from krishiv.sql import _rowconv
from krishiv.sql.conf import RuntimeConfig
from krishiv.sql.dataframe import DataFrame
from krishiv.sql.readwriter import DataFrameReader
from krishiv.sql.types import DataType, StructField, StructType
from krishiv.storagelevel import StorageLevel

if TYPE_CHECKING:
    from krishiv.sql.udf import UDFRegistration

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
                    _active = SparkSession(_native.Session(list(self._config.items())), list(self._config.items()))
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
            return SparkSession(_native.Session(list(self._config.items())), list(self._config.items()))

    class _BuilderProperty:
        """``SparkSession.builder``: a fresh Builder per access, so options do not leak between callers."""

        def __get__(self, instance: object, owner: type | None = None) -> SparkSession.Builder:
            return SparkSession.Builder()

    # Typed as Builder (what attribute access gives); the descriptor below provides it.
    builder: Builder = _BuilderProperty()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    def __init__(self, native: _native.Session, config: list[tuple[str, str]] | None = None) -> None:
        self._native = native
        self._conf = RuntimeConfig(native)
        # The options the session was built with: `newSession` starts from them.
        self._config = list(config or [])

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
    def udf(self) -> UDFRegistration:
        """The :class:`UDFRegistration` of this session (``spark.udf.register(name, f, returnType)``)."""
        from krishiv.sql.udf import UDFRegistration

        return UDFRegistration(self)

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
        ids = range(start, end, step)
        chunk = 1 << 20
        table = pa.table(
            {"id": pa.chunked_array([ids[i : i + chunk] for i in range(0, len(ids), chunk)], type=pa.int64())}
        )
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
        a list of column names (types are inferred from the rows as PySpark does:
        each value by its Python type, the rows merged, so an ``int`` and a ``float`` in
        one column raise ``CANNOT_MERGE_TYPE``), or a single atomic type (values become
        one ``value`` column). With ``verifySchema``
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
            from krishiv.sql import _convert

            pa = _convert.require_pyarrow("createDataFrame")
            return self._from_arrow(pa.table(data), schema, pa)
        rows = list(data)
        if schema is None or isinstance(schema, list | tuple):
            return self._create_from_rows(rows, schema, verifySchema)
        return self._from_rows(rows, schema, verifySchema, Row, null_on_error=not verifySchema)

    def _from_pandas(self, data: Any, schema: Any) -> DataFrame:
        from krishiv.sql import _convert

        pa = _convert.require_pyarrow("createDataFrame")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise PySparkValueError(errorClass="CANNOT_INFER_EMPTY_SCHEMA")
        table = pa.Table.from_pandas(data, preserve_index=False)
        # Column labels that are not strings (``0``, ``1``) are named by their text.
        table = table.rename_columns([str(n) for n in table.column_names])
        return self._from_arrow(table, schema, pa)

    def _from_arrow(self, table: Any, schema: Any, pa: Any) -> DataFrame:
        """A DataFrame of an Arrow table: its columns taken as Spark types (see
        :func:`krishiv.sql._rowconv.normalize_arrow_table`), then renamed by a list of
        names or cast to an explicit schema."""
        ntz = (self._conf.get("spark.sql.timestampType", "TIMESTAMP_LTZ") or "").strip().upper() == "TIMESTAMP_NTZ"
        table = _rowconv.normalize_arrow_table(table, pa, None if ntz else _session_zone(self))
        if isinstance(schema, list | tuple):
            _refuse_duplicate_names([str(n) for n in schema])
        elif schema is None:
            _refuse_duplicate_names(table.column_names)
        if schema is None:
            return DataFrame(self._native.create_dataframe(table), self)
        if isinstance(schema, list | tuple):
            if len(schema) != table.num_columns or not all(isinstance(n, str) for n in schema):
                raise not_expected_type("schema", "list of str", schema)
            return DataFrame(self._native.create_dataframe(table.rename_columns(list(schema))), self)
        struct = self._explicit_schema(schema)
        try:
            return DataFrame(self._native.create_dataframe(table.cast(_pa_schema(struct, pa))), self)
        except pa.ArrowNotImplementedError:
            # e.g. a column of dicts for a ``map`` (Arrow reads them as structs): through Python values.
            from krishiv.sql.row import Row

            rows = [tuple(r.values()) for r in table.to_pylist()]
            return self._from_rows(rows, struct, False, Row)

    def _from_rows(
        self, rows: list[Any], schema: Any, verify: bool, Row: Any, null_on_error: bool = False
    ) -> DataFrame:
        from krishiv.sql import _convert

        pa = _convert.require_pyarrow("createDataFrame")
        struct = self._explicit_schema(schema)
        if not _is_struct_schema(schema):
            rows = [(row,) for row in rows]  # an atomic schema: one ``value`` column, one value per row
        elif verify:
            for row in rows:
                if not isinstance(row, dict | list | tuple):
                    raise _rowconv.unacceptable(None, struct, row)
        named = [f.name for f in struct.fields]
        _refuse_duplicate_names(named)
        width = len(named)
        valued = [_row_values(row, named, Row) for row in rows]
        if verify:
            for values in valued:
                _rowconv.verify(values, struct)
        for values in valued:
            if len(values) != width:
                raise _axis_length_mismatch(width, len(values))
        zone = lambda: _convert.session_zone(_session_zone(self))  # noqa: E731
        columns = []
        for i, field in enumerate(struct.fields):
            arrow_type = _rowconv.pa_type(field.dataType, pa)
            values = [_rowconv.convert(v[i], field.dataType, zone) for v in valued]
            columns.append(_rowconv.arrow_array(values, arrow_type, pa, null_on_error))
        table = pa.table(columns, schema=_pa_schema(struct, pa))
        return DataFrame(self._native.create_dataframe(table), self)

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
        as ``pyspark.sql.SparkSession.newSession``: it starts from the options
        this session was built with, not from later ``conf.set`` changes."""
        return SparkSession(_native.Session(list(self._config)), self._config)

    def _create_from_rows(
        self, data: list[Any], schema: list[Any] | tuple[Any, ...] | None, verify: bool = True
    ) -> DataFrame:
        """A DataFrame of Python rows with column names, or none: names come from ``schema``,
        else the first ``Row`` or dict, else ``_1``, ``_2``, ...; the types are PySpark's
        inference over every row (see :mod:`krishiv.sql._infer`), converted like an explicit schema."""
        from functools import reduce

        from krishiv.sql._infer import has_nulltype, infer_row, merge_type
        from krishiv.sql.row import Row

        names: list[str] | None = None
        if schema is not None:
            names = list(schema)
            for name in names:
                if not isinstance(name, str):
                    raise not_expected_type("schema", "list of str", name)
        for row in data:
            if not isinstance(row, Row | tuple | list | dict):
                raise PySparkTypeError(
                    errorClass="CANNOT_INFER_SCHEMA_FOR_TYPE",
                    messageParameters={"data_type": type(row).__name__},
                    sqlState="42K09",
                )
        if not data:
            raise PySparkValueError(errorClass="CANNOT_INFER_EMPTY_SCHEMA")
        timestamp_type = self._conf.get("spark.sql.timestampType", "TIMESTAMP_LTZ") or "TIMESTAMP_LTZ"
        timestamp_ntz = timestamp_type.strip().upper() == "TIMESTAMP_NTZ"
        # PySpark's ``_infer_schema`` per row (a dict by its sorted keys, a ``Row`` or named tuple by its
        # fields, any other tuple by ``names`` then ``_1``, ``_2``, ...), the rows merged as ``_merge_type``.
        inferred = []
        for row in data:
            if isinstance(row, dict):
                row_names = sorted(row)
                values = [row[k] for k in row_names]
            else:
                values = list(row)
                row_names = list(getattr(row, "__fields__", None) or getattr(row, "_fields", None) or [])
                if not row_names:
                    given = (names or [])[: len(values)]
                    row_names = given + [f"_{i + 1}" for i in range(len(given), len(values))]
            inferred.append(infer_row(values, row_names, timestamp_ntz))
        merged = reduce(merge_type, inferred)
        if not isinstance(merged, StructType) or has_nulltype(merged):
            raise PySparkValueError(errorClass="CANNOT_DETERMINE_TYPE")
        # The merged row type is renamed by position: fewer names keep the inferred ones for the rest,
        # more names fail.
        pre = [f.name for f in merged.fields]
        width = len(pre)
        final = pre
        if names is not None:
            if len(names) > width:
                raise _axis_length_mismatch(len(names), width)
            final = names + pre[len(names) :]
        valued: list[list[Any]] = []
        for row in data:
            if isinstance(row, dict):
                valued.append([row.get(name) for name in pre])
                continue
            if len(row) != width:
                if hasattr(row, "__fields__"):
                    raise PySparkValueError(
                        errorClass="STRUCT_ARRAY_LENGTH_MISMATCH",
                        messageParameters={"expected": str(width), "actual": str(len(row))},
                        sqlState="2201E",
                    )
                raise _axis_length_mismatch(width, len(row))
            valued.append(list(row))
        struct = StructType([StructField(n, f.dataType, f.nullable) for n, f in zip(final, merged.fields, strict=True)])
        return self._from_rows(valued, struct, False, Row, null_on_error=not verify)

    def stop(self) -> None:
        """Ends this session; the next ``getOrCreate`` makes a new one."""
        global _active
        with _lock:
            if _active is self:
                _active = None


def _is_pandas(data: Any) -> bool:
    # A pandas DataFrame means pandas is already imported; never import it here.
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(data, pd.DataFrame)


_pa_type = _rowconv.pa_type


def _pa_schema(struct: StructType, pa: Any) -> Any:
    return _rowconv.pa_schema(struct, pa)


def _refuse_duplicate_names(names: list[str]) -> None:
    """The engine reads a registered table's columns by name, so duplicate names (which Spark
    allows) are refused up front instead of failing at the first action."""
    seen: set[str] = set()
    for name in names:
        if name.lower() in seen:
            raise unsupported_feature(f"createDataFrame with the duplicate column name `{name}`")
        seen.add(name.lower())


def _is_struct_schema(schema: Any) -> bool:
    """Whether ``schema`` (a ``StructType`` or a DDL string) describes rows rather than one value."""
    if isinstance(schema, StructType):
        return True
    if isinstance(schema, str):
        return isinstance(DataType.fromDDL(schema), StructType)
    return False


def _row_values(row: Any, names: list[str] | None, Row: Any) -> list[Any]:
    """One row of ``createDataFrame`` data as its values: a tuple, list or ``Row`` (named or
    not) in order, as PySpark converts rows against an explicit schema; a dict by field name
    (missing keys are NULL); any other value is a single-column row."""
    if isinstance(row, Row):
        return list(row)
    if isinstance(row, dict):
        return list(row.values()) if names is None else [row.get(n) for n in names]
    if isinstance(row, list | tuple):
        return list(row)
    return [row]


def _axis_length_mismatch(expected: int, actual: int) -> PySparkValueError:
    """PySpark's error for rows whose length differs from the schema's."""
    return PySparkValueError(
        errorClass="AXIS_LENGTH_MISMATCH",
        messageParameters={"expected_length": str(expected), "actual_length": str(actual)},
    )


def _session_zone(session: SparkSession) -> str:
    return session.conf.get("spark.sql.session.timeZone", "UTC") or "UTC"


class Catalog:
    """The catalog of temporary views, as ``pyspark.sql.catalog.Catalog``.

    The embedded engine has no databases or persistent tables: every table
    is a temporary view in the ``default`` database.
    """

    def __init__(self, session: SparkSession) -> None:
        self._session = session

    def _user_tables(self) -> list[str]:
        """The temporary tables and views, without those holding ``createDataFrame`` data."""
        return [n for n in self._session._native.temp_table_names() if not _native.Session.is_dataframe_table(n)]

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
        names = self._user_tables()
        return any(n.lower() == tableName.lower() for n in names)

    def listTables(self, dbName: str | None = None) -> DataFrame:
        """The temporary views as a DataFrame of ``(name, database,
        description, tableType, isTemporary)`` rows.

        .. note:: The engine has no catalogs or namespaces, so this is the
            classic five-column shape rather than Spark 4.x's namespaced one.
        """
        if dbName is not None and not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        rows = [(n, "default", None, "TEMPORARY", True) for n in self._user_tables()]
        return self._session.createDataFrame(
            rows,
            "name string, database string, description string, tableType string, isTemporary boolean",
        )

    def dropTempView(self, viewName: str) -> bool:
        """Drops the temporary view ``viewName``; whether it existed."""
        if not isinstance(viewName, str):
            raise not_expected_type("viewName", "str", viewName)
        return self._session._native.drop_temp_view(viewName)

    def registerFunction(self, name: str, f: Any, returnType: Any = None) -> Any:
        """An alias for ``spark.udf.register`` (deprecated since 2.3.0, ``pyspark/sql/catalog.py:1272``)."""
        warnings.warn("Deprecated in 2.3.0. Use spark.udf.register instead.", FutureWarning, stacklevel=2)
        return self._session.udf.register(name, f, returnType)

    def clearCache(self) -> None:
        """Drops every cached relation: later actions run uncached."""
        self._session._native.clear_cache()

    def uncacheTable(self, tableName: str) -> None:
        """Drops the cached relations reading ``tableName``."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        self._session._native.uncache_table(tableName)

    def refreshTable(self, tableName: str) -> DataFrame:
        """Drops the cached relations reading ``tableName``, so the next action reads the
        table again, and returns the table."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        self._session._native.refresh_table(tableName)
        return self._session.table(tableName)

    def refreshByPath(self, path: str) -> None:
        """Drops the cached relations reading files by path, so the next action lists the
        files again."""
        if not isinstance(path, str):
            raise not_expected_type("path", "str", path)
        self._session._native.refresh_files()

    def isCached(self, tableName: str) -> bool:
        """Whether the table ``tableName`` is cached now."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        try:
            return self._session.table(tableName).is_cached
        except AnalysisException:
            return False

    def cacheTable(self, tableName: str, storageLevel: StorageLevel | None = None) -> None:
        """Caches ``tableName`` at ``storageLevel`` (default ``MEMORY_AND_DISK``), eagerly:
        the table is read now."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        level = storageLevel if storageLevel is not None else StorageLevel.MEMORY_AND_DISK
        if not isinstance(level, StorageLevel):
            raise not_expected_type("storageLevel", "StorageLevel", storageLevel)
        self._session.table(tableName).persist(level).count()
