# SPDX-License-Identifier: Apache-2.0
"""``SparkSession``: the entry point, as ``pyspark.sql.SparkSession`` (embedded: no server)."""

from __future__ import annotations

import threading
import warnings
from typing import Any

from krishiv import _native
from krishiv.errors import not_expected_type, not_implemented
from krishiv.sql.conf import RuntimeConfig
from krishiv.sql.dataframe import DataFrame
from krishiv.sql.readwriter import DataFrameReader

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
        (a pyarrow Table, a Polars or pandas-via-pyarrow frame, a DuckDB result, ...)."""
        if schema is not None:
            raise not_implemented("createDataFrame with a schema")
        if not (hasattr(data, "__arrow_c_stream__") or hasattr(data, "__arrow_c_array__")):
            raise not_implemented("createDataFrame from rows (only Arrow-producing objects are supported)")
        return DataFrame(self._native.create_dataframe(data), self)

    def stop(self) -> None:
        """Ends this session; the next ``getOrCreate`` makes a new one."""
        global _active
        with _lock:
            if _active is self:
                _active = None
