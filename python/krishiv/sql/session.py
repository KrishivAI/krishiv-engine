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
    PySparkRuntimeError,
    PySparkTypeError,
    PySparkValueError,
    not_expected_type,
    not_implemented,
    not_implemented_attribute,
    unsupported_attribute,
    unsupported_feature,
)
from krishiv.sql import _rowconv
from krishiv.sql.catalog import CatalogMetadata, Column, Database, Function, Table
from krishiv.sql.conf import RuntimeConfig, _warn_if_cluster_setting
from krishiv.sql.dataframe import DataFrame
from krishiv.sql.readwriter import DataFrameReader
from krishiv.sql.types import DataType, StructField, StructType, TimestampNTZType, TimestampType
from krishiv.storagelevel import StorageLevel

# The class's `_native` property hides the module in annotations written inside it.
_NativeSession = _native.Session

if TYPE_CHECKING:
    from krishiv._console import Console
    from krishiv._udf.profiler import Profile
    from krishiv.sql.datasource import DataSourceRegistration
    from krishiv.sql.udf import UDFRegistration

#: The Spark version this engine is compatible with.
SPARK_VERSION = "4.3.0"


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
                    self._put(k, v)
            if key is not None:
                self._put(key, value)
            return self

        def _put(self, key: str, value: Any) -> None:
            """Stores one option. Every public setter calls this directly, so the cluster-setting
            warning is raised at the same depth (the user's line) for each of them."""
            self._config[key] = ("true" if value else "false") if isinstance(value, bool) else str(value)
            _warn_if_cluster_setting(key, self._config[key], stacklevel=3)

        def appName(self, name: str) -> SparkSession.Builder:
            self._put("spark.app.name", name)
            return self

        def master(self, master: str) -> SparkSession.Builder:
            self._put("spark.master", master)
            return self

        def enableHiveSupport(self) -> SparkSession.Builder:
            """Not supported: there is no Hive metastore in Krishiv (``KRISHIV_UNSUPPORTED_FEATURE``)."""
            raise unsupported_feature("SparkSession.Builder.enableHiveSupport (there is no Hive metastore)")

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
                            _active.conf._set(k, v)  # the builder warned when the option was given
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

    def __init__(self, native: _NativeSession, config: list[tuple[str, str]] | None = None) -> None:
        self._base_native = native
        # `addTag`'s tags are per thread, as in Spark: each thread's frames carry its own.
        self._tag_state = threading.local()
        self._conf = RuntimeConfig(native, self)
        from krishiv._udf.profiler import Registry

        self._profile_registry = Registry()
        # The options the session was built with: `newSession` starts from them.
        self._config = list(config or [])
        # The session's web console once `krishiv.ui()` has started one; `stop` stops it.
        self._console: Console | None = None
        # Held to read and to change `_console`, here and by `krishiv.ui()`.
        self._console_lock = threading.Lock()

    @property
    def _native(self) -> _NativeSession:
        """The engine session: a handle carrying this thread's tags (``addTag``) when it has any."""
        tags = self.getTags()
        if not tags:
            return self._base_native
        key = tuple(sorted(tags))
        cached = getattr(self._tag_state, "handle", None)
        if cached is None or cached[0] != key:
            cached = (key, self._base_native.tagged(list(key)))
            self._tag_state.handle = cached
        return cached[1]

    @property
    def version(self) -> str:
        return SPARK_VERSION

    def interruptAll(self) -> list[str]:
        """Cancels every running query of this session; the operation ids it cancelled."""
        return self._base_native.interrupt_all()

    def interruptTag(self, tag: str) -> list[str]:
        """Cancels the running queries that carry ``tag`` (see :meth:`addTag`); their operation ids."""
        # As Spark Connect: a tag no query carries (an invalid one included) cancels nothing.
        return self._base_native.interrupt_tag(tag) if isinstance(tag, str) else []

    def interruptOperation(self, op_id: str) -> list[str]:
        """Cancels the running query with operation id ``op_id``; its id in a list, or an empty list."""
        if not isinstance(op_id, str):
            raise not_expected_type("op_id", "str", op_id)
        return self._base_native.interrupt_operation(op_id)

    def addTag(self, tag: str) -> None:
        """Tags the queries this thread starts from now on, until :meth:`removeTag`; :meth:`interruptTag` cancels them.

        A frame keeps the tags it was made with: tag before the ``spark.sql``, ``spark.table``, ``spark.read``
        or ``spark.range`` call, not before the action.
        """
        self._check_tag(tag)
        self._tags().add(tag)

    def removeTag(self, tag: str) -> None:
        """Stops tagging with ``tag`` (no error when this thread has not added it)."""
        self._check_tag(tag)
        self._tags().discard(tag)

    def getTags(self) -> set[str]:
        """This thread's tags."""
        return self._tags()

    def clearTags(self) -> None:
        """Removes every tag of this thread."""
        self._tags().clear()

    def _tags(self) -> set[str]:
        tags = getattr(self._tag_state, "tags", None)
        if tags is None:
            tags = self._tag_state.tags = set()
        return tags

    @staticmethod
    def _check_tag(tag: str) -> None:
        if tag is None:
            raise PySparkValueError(errorClass="CANNOT_BE_NONE", messageParameters={"arg_name": "Spark Connect tag"})
        if not isinstance(tag, str):
            raise not_expected_type("tag", "str", tag)
        if "," in tag:
            raise PySparkValueError(
                errorClass="VALUE_ALLOWED",
                messageParameters={"arg_name": "Spark Connect tag", "disallowed_value": ","},
            )
        if not tag:
            raise PySparkValueError(
                errorClass="VALUE_NOT_NON_EMPTY_STR",
                messageParameters={"arg_name": "Spark Connect tag", "arg_value": tag},
            )

    @property
    def sparkContext(self) -> Any:
        """Not supported: Krishiv has no SparkContext or RDDs; use the DataFrame API."""
        raise unsupported_attribute("SparkSession.sparkContext (SparkContext and RDDs; use the DataFrame API)")

    @property
    def conf(self) -> RuntimeConfig:
        return self._conf

    @property
    def profile(self) -> Profile:
        """The UDF profiles of this session (``spark.sql.pyspark.udf.profiler`` = ``perf`` or ``memory``):
        ``show``, ``dump`` and ``clear``, as ``pyspark.sql.profiler.Profile``."""
        from krishiv._udf.profiler import Profile

        return Profile(self._profile_registry)

    @property
    def read(self) -> DataFrameReader:
        return DataFrameReader(self)

    @property
    def dataSource(self) -> DataSourceRegistration:
        """``spark.dataSource.register(cls)``: registers a Python data source class (see
        :mod:`krishiv.sql.datasource`)."""
        from krishiv.sql.datasource import DataSourceRegistration

        return DataSourceRegistration(self)

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
        explicit = schema is not None and not isinstance(schema, list | tuple)
        if (data.shape[0] == 0 and not explicit) or data.shape[1] == 0:
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
        # With ``spark.sql.timestampType=TIMESTAMP_NTZ`` a naive column is inferred as wall-clock time, but
        # an explicit ``TimestampType`` target localizes it in the session zone (and an explicit NTZ one never).
        wall_clock: frozenset[int] = frozenset(range(table.num_columns)) if ntz else frozenset()
        if schema is not None and not isinstance(schema, list | tuple):
            targets = self._explicit_schema(schema).fields
            wall_clock |= frozenset(i for i, f in enumerate(targets) if isinstance(f.dataType, TimestampNTZType))
            wall_clock -= frozenset(i for i, f in enumerate(targets) if isinstance(f.dataType, TimestampType))
        if schema is not None and not isinstance(schema, list | tuple):
            # An explicit target is applied to an unsigned source column first (Spark casts to the target's Arrow
            # type without inferring the source); with no target an unsigned column is refused.
            struct = self._explicit_schema(schema)
            if len(struct.fields) == table.num_columns:
                target = _pa_schema(struct, pa)
                table = pa.table(
                    [
                        c.cast(target.field(i).type) if pa.types.is_unsigned_integer(c.type) else c
                        for i, c in enumerate(table.columns)
                    ],
                    names=table.column_names,
                )
        table = _rowconv.normalize_arrow_table(table, pa, _session_zone(self), wall_clock)
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
        if len(struct.fields) == table.num_columns:
            table = table.rename_columns([f.name for f in struct.fields])
        try:
            return DataFrame(self._native.create_dataframe(table.cast(_pa_schema(struct, pa))), self)
        except pa.ArrowNotImplementedError:
            # e.g. a column of dicts for a ``map`` (Arrow reads them as structs): through Python values.
            from krishiv.sql.row import Row

            columns = [[_arrow_map_to_dict(v, column.type, pa) for v in column.to_pylist()] for column in table.columns]
            rows = list(zip(*columns, strict=True)) if columns else [() for _ in range(table.num_rows)]
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
            values = []
            for v in valued:
                try:
                    values.append(_rowconv.convert(v[i], field.dataType, zone))
                except (TypeError, ValueError, OverflowError):
                    # An unexpected container (``42`` for an array) is NULL when verification is off, as the
                    # JVM conversion of classic PySpark leaves it.
                    if not null_on_error:
                        raise
                    values.append(None)
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
         |-- age: integer (nullable = true)
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
        inferred: list[DataType] = []
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

    @classmethod
    def getActiveSession(cls) -> SparkSession | None:
        """The session ``getOrCreate`` made and ``stop`` has not ended, or ``None``."""
        with _lock:
            return _active

    @classmethod
    def active(cls) -> SparkSession:
        """The active session; fails with ``NO_ACTIVE_OR_DEFAULT_SESSION`` when there is none, as PySpark."""
        session = cls.getActiveSession()
        if session is None:
            raise PySparkRuntimeError(errorClass="NO_ACTIVE_OR_DEFAULT_SESSION", messageParameters={})
        return session

    @property
    def readStream(self) -> Any:
        """Not available: streaming reads are not part of 0.2 (``NOT_IMPLEMENTED``)."""
        raise not_implemented_attribute("SparkSession.readStream (streaming)")

    @property
    def streams(self) -> Any:
        """Not available: streaming queries are not part of 0.2 (``NOT_IMPLEMENTED``)."""
        raise not_implemented_attribute("SparkSession.streams (streaming)")

    @property
    def tvf(self) -> Any:
        """Not available: the ``TableValuedFunction`` namespace (``NOT_IMPLEMENTED``)."""
        raise not_implemented_attribute("SparkSession.tvf")

    @property
    def udtf(self) -> Any:
        """Not available: user-defined table functions (``NOT_IMPLEMENTED``)."""
        raise not_implemented_attribute("SparkSession.udtf")

    def addArtifacts(self, *path: str, pyfile: bool = False, archive: bool = False, file: bool = False) -> None:
        """Not available: artifacts belong to Spark Connect sessions (``NOT_IMPLEMENTED``)."""
        raise not_implemented("SparkSession.addArtifacts")

    addArtifact = addArtifacts

    def copyFromLocalToFs(self, local_path: str, dest_path: str) -> None:
        """Not available: artifacts belong to Spark Connect sessions (``NOT_IMPLEMENTED``)."""
        raise not_implemented("SparkSession.copyFromLocalToFs")

    def stop(self) -> None:
        """Ends this session; the next ``getOrCreate`` makes a new one."""
        global _active
        with _lock:
            if _active is self:
                _active = None
        with self._console_lock:
            console, self._console = self._console, None
        if console is not None:
            console.stop()


def _is_pandas(data: Any) -> bool:
    # A pandas DataFrame means pandas is already imported; never import it here.
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(data, pd.DataFrame)


_pa_type = _rowconv.pa_type


def _arrow_map_to_dict(value: Any, arrow_type: Any, pa: Any) -> Any:
    """``value`` (from ``to_pylist``) with each Arrow map, which Python reads as a list of key/value pairs, turned
    into a ``dict``, at any depth; nulls stay ``None``."""
    if value is None:
        return None
    if pa.types.is_map(arrow_type):
        return {
            _arrow_map_to_dict(k, arrow_type.key_type, pa): _arrow_map_to_dict(v, arrow_type.item_type, pa)
            for k, v in value
        }
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return [_arrow_map_to_dict(v, arrow_type.value_type, pa) for v in value]
    if pa.types.is_struct(arrow_type):
        return {f.name: _arrow_map_to_dict(value[f.name], f.type, pa) for f in arrow_type}
    return value


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


def _qualified(name: str, db: str | None) -> str:
    """``db.name`` when a database is given, as PySpark joins ``dbName`` and ``tableName``."""
    return name if db is None else f"{db}.{name}"


class Catalog:
    """The catalog API, as ``pyspark.sql.catalog.Catalog``; the lists are Python lists of named tuples.

    ``spark_catalog`` is the only catalog this API lists or accepts; its ``default`` database holds the
    temporary views and the tables ``CREATE TABLE`` made.
    """

    def __init__(self, session: SparkSession) -> None:
        self._session = session

    @property
    def _n(self) -> Any:
        return self._session._native

    def currentCatalog(self) -> str:
        """The current catalog's name."""
        return str(self._n.catalog_current_catalog())

    def setCurrentCatalog(self, catalogName: str) -> None:
        """Makes ``catalogName`` the current catalog; only ``spark_catalog`` is accepted.

        SQL ``USE system`` still reaches the system tables.
        """
        if not isinstance(catalogName, str):
            raise not_expected_type("catalogName", "str", catalogName)
        self._n.catalog_set_current_catalog(catalogName)

    def listCatalogs(self, pattern: str | None = None) -> list[CatalogMetadata]:
        """The catalogs, filtered by the ``|``-separated glob ``pattern``.

        Only ``spark_catalog`` is listed; the system tables stay reachable through SQL ``USE system``.
        """
        if pattern is not None and not isinstance(pattern, str):
            raise not_expected_type("pattern", "str", pattern)
        names = [str(n) for n in self._n.catalog_list_catalogs()]
        if pattern is not None:
            names = [n for n in names if _glob_match(pattern, n)]
        return [CatalogMetadata(name=n, description=None) for n in names]

    def currentDatabase(self) -> str:
        """The current database's name."""
        return str(self._n.catalog_current_database())

    def setCurrentDatabase(self, dbName: str) -> None:
        """Makes ``dbName`` the current database."""
        if not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        self._n.catalog_set_current_database(dbName)

    def listDatabases(self, pattern: str | None = None) -> list[Database]:
        """The databases, filtered by the ``|``-separated glob ``pattern``."""
        if pattern is not None and not isinstance(pattern, str):
            raise not_expected_type("pattern", "str", pattern)
        return [Database(*row) for row in self._n.catalog_list_databases(pattern)]

    def getDatabase(self, dbName: str) -> Database:
        """The database ``dbName``; ``SCHEMA_NOT_FOUND`` when there is none."""
        if not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        return Database(*self._n.catalog_get_database(dbName))

    def databaseExists(self, dbName: str) -> bool:
        """Whether the database ``dbName`` exists."""
        if not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        return bool(self._n.catalog_database_exists(dbName))

    def listTables(self, dbName: str | None = None, pattern: str | None = None) -> list[Table]:
        """The tables and views of ``dbName`` (default: the current database), filtered by ``pattern``.

        Local temporary views are included in the current database, as in Spark; the tables that hold
        ``createDataFrame`` data are not.
        """
        if dbName is not None and not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        if pattern is not None and not isinstance(pattern, str):
            raise not_expected_type("pattern", "str", pattern)
        rows = self._n.catalog_list_tables(dbName, pattern)
        return [Table(*row) for row in rows if not _native.Session.is_dataframe_table(row[0])]

    def getTable(self, tableName: str) -> Table:
        """The table or view ``tableName``; ``TABLE_OR_VIEW_NOT_FOUND`` when there is none."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        _refuse_hidden_table(tableName)
        return Table(*self._n.catalog_get_table(tableName))

    def tableExists(self, tableName: str, dbName: str | None = None) -> bool:
        """Whether the table or view ``tableName`` exists (in ``dbName`` when given)."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        if dbName is not None and not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        if _is_hidden_table(tableName):
            return False
        return bool(self._n.catalog_table_exists(_qualified(tableName, dbName)))

    def listColumns(self, tableName: str, dbName: str | None = None) -> list[Column]:
        """The columns of ``tableName`` (in ``dbName`` when given)."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        if dbName is not None and not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        _refuse_hidden_table(tableName)
        return [Column(*row) for row in self._n.catalog_list_columns(_qualified(tableName, dbName))]

    def listFunctions(self, dbName: str | None = None, pattern: str | None = None) -> list[Function]:
        """The functions, filtered by ``pattern``.

        .. note:: Functions live in no database here; a ``dbName`` must name an existing database and
            does not narrow the list.
        """
        if dbName is not None:
            if not isinstance(dbName, str):
                raise not_expected_type("dbName", "str", dbName)
            self._n.catalog_get_database(dbName)
        if pattern is not None and not isinstance(pattern, str):
            raise not_expected_type("pattern", "str", pattern)
        return [Function(*row) for row in self._n.catalog_list_functions(pattern)]

    def getFunction(self, functionName: str) -> Function:
        """The function ``functionName``; ``UNRESOLVED_ROUTINE`` when there is none."""
        if not isinstance(functionName, str):
            raise not_expected_type("functionName", "str", functionName)
        return Function(*self._n.catalog_get_function(functionName))

    def functionExists(self, functionName: str, dbName: str | None = None) -> bool:
        """Whether the function ``functionName`` exists."""
        if not isinstance(functionName, str):
            raise not_expected_type("functionName", "str", functionName)
        if dbName is not None and not isinstance(dbName, str):
            raise not_expected_type("dbName", "str", dbName)
        return bool(self._n.catalog_function_exists(_qualified(functionName, dbName)))

    def dropTempView(self, viewName: str) -> bool:
        """Drops the temporary view ``viewName``; whether it existed."""
        if not isinstance(viewName, str):
            raise not_expected_type("viewName", "str", viewName)
        return self._session._native.drop_temp_view(viewName)

    def dropGlobalTempView(self, viewName: str) -> bool:
        """Drops the global temporary view ``viewName`` (of ``global_temp``); whether it existed."""
        if not isinstance(viewName, str):
            raise not_expected_type("viewName", "str", viewName)
        qualified = f"global_temp.{viewName}"
        if not self._session.catalog.tableExists(qualified):
            return False
        self._session.sql(f"DROP VIEW {qualified}")
        return True

    def createTable(
        self,
        tableName: str,
        path: str | None = None,
        source: str | None = None,
        schema: StructType | None = None,
        description: str | None = None,
        **options: str,
    ) -> DataFrame:
        """Creates a table over the data at ``path`` (an external table) or, with no ``path``, an empty managed
        one, and returns it as a DataFrame. ``source`` defaults to ``spark.sql.sources.default``; ``schema``
        gives the columns, else the source infers them. An existing table fails as ``CREATE TABLE`` does."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        if path is not None and not isinstance(path, str):
            raise not_expected_type("path", "str", path)
        if source is not None and not isinstance(source, str):
            raise not_expected_type("source", "str", source)
        if schema is not None and not isinstance(schema, StructType):
            raise not_expected_type("schema", "StructType", schema)
        if description is not None and not isinstance(description, str):
            raise not_expected_type("description", "str", description)

        def quote(text: str) -> str:
            return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"

        fmt = source or self._session.conf.get("spark.sql.sources.default", "parquet")
        columns = ""
        if schema is not None:
            cols = ", ".join(
                "`" + f.name.replace("`", "``") + "` " + f.dataType.simpleString() + ("" if f.nullable else " NOT NULL")
                for f in schema.fields
            )
            columns = f" ({cols})"
        opts = ""
        if options:
            opts = " OPTIONS (" + ", ".join(f"{quote(str(k))} {quote(str(v))}" for k, v in options.items()) + ")"
        location = f" LOCATION {quote(path)}" if path is not None else ""
        comment = f" COMMENT {quote(description)}" if description is not None else ""
        self._session.sql(f"CREATE TABLE {tableName}{columns} USING {fmt}{opts}{comment}{location}")
        return self._session.table(tableName)

    def createExternalTable(
        self,
        tableName: str,
        path: str | None = None,
        source: str | None = None,
        schema: StructType | None = None,
        **options: str,
    ) -> DataFrame:
        """Deprecated alias of :meth:`createTable` (deprecated since Spark 2.2)."""
        warnings.warn(
            "createExternalTable is deprecated since Spark 2.2, use createTable instead.", FutureWarning, stacklevel=2
        )
        return self.createTable(tableName, path, source, schema, None, **options)

    def recoverPartitions(self, tableName: str) -> None:
        """Recovers all the partitions of the table found in its location (``MSCK REPAIR TABLE``)."""
        if not isinstance(tableName, str):
            raise not_expected_type("tableName", "str", tableName)
        self._session.sql(f"MSCK REPAIR TABLE {tableName}")

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


def _is_hidden_table(name: str) -> bool:
    """Whether ``name`` is one of the tables that hold ``createDataFrame`` data (listed nowhere)."""
    return bool(_native.Session.is_dataframe_table(_last_part(name)))


def _last_part(name: str) -> str:
    """The last dot-separated part of ``name``; a backquoted part may hold dots (a doubled backquote is one)."""
    parts: list[str] = []
    cur: list[str] = []
    quoted = False
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == "`":
            if quoted and name[i + 1 : i + 2] == "`":
                cur.append("`")
                i += 1
            else:
                quoted = not quoted
        elif ch == "." and not quoted:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts[-1]


def _refuse_hidden_table(name: str) -> None:
    """``TABLE_OR_VIEW_NOT_FOUND`` for a ``createDataFrame`` table, as for any table the catalog does not show."""
    if _is_hidden_table(name):
        raise AnalysisException(
            errorClass="TABLE_OR_VIEW_NOT_FOUND",
            messageParameters={"relationName": f"`{name}`"},
            sqlState="42P01",
        )


def _glob_match(pattern: str, name: str) -> bool:
    """Spark's ``|``-separated, ``*``-wildcard, case-insensitive name pattern."""
    import fnmatch

    return any(fnmatch.fnmatchcase(name.lower(), p.strip().lower()) for p in pattern.split("|"))
