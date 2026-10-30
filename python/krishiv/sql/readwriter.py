# SPDX-License-Identifier: Apache-2.0
"""``DataFrameReader`` (``spark.read``), ``DataFrameWriter`` (``df.write``)
and ``DataFrameWriterV2`` (``df.writeTo``).

The format and path are checked when the query is analyzed; save modes are
checked by the engine (``SaveMode::parse``), so the writer passes the mode
string through unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from krishiv.errors import not_expected_type
from krishiv.sql.column import Column
from krishiv.sql.dataframe import DataFrame

if TYPE_CHECKING:
    from krishiv.sql.session import SparkSession


class DataFrameReader:
    def __init__(self, session: SparkSession) -> None:
        self._session = session
        self._format = "parquet"
        self._options: list[tuple[str, str]] = []

    def format(self, source: str) -> DataFrameReader:
        self._format = source
        return self

    def option(self, key: str, value: object) -> DataFrameReader:
        self._options.append((key, _option_text(value)))
        return self

    def options(self, **options: object) -> DataFrameReader:
        for key, value in options.items():
            self.option(key, value)
        return self

    def load(self, path: str | list[str] | None = None, format: str | None = None, **options: object) -> DataFrame:
        if format is not None:
            self.format(format)
        self.options(**options)
        if path is None:
            raise not_expected_type("path", "str or list of str", path)
        paths = [path] if isinstance(path, str) else list(path)
        return DataFrame(self._session._native.read(self._format, paths, self._options), self._session)

    def parquet(self, *paths: str) -> DataFrame:
        return self.load(list(paths), format="parquet")

    def csv(self, path: str | list[str], **options: object) -> DataFrame:
        return self.load(path, format="csv", **options)

    def json(self, path: str | list[str], **options: object) -> DataFrame:
        return self.load(path, format="json", **options)


class DataFrameWriter:
    """Writes a :class:`DataFrame`'s rows to files (a subset of
    ``pyspark.sql.DataFrameWriter``: ``format``/``mode``/``option``/
    ``options``/``partitionBy``/``save`` and the ``parquet``/``csv``/``json``
    shortcuts; ``bucketBy``/``sortBy``/``saveAsTable``/``insertInto`` are not
    implemented). The engine writes no partitioned layout, so a write with
    ``partitionBy`` columns always fails with ``KRISHIV_UNSUPPORTED_FEATURE``;
    unknown formats fail the same way and unknown modes with
    ``INVALID_SAVE_MODE``."""

    def __init__(self, df: DataFrame) -> None:
        self._df = df
        self._format = "parquet"
        self._mode = "error"
        self._options: list[tuple[str, str]] = []
        self._partition_by: list[str] = []

    def format(self, source: str) -> DataFrameWriter:
        self._format = source
        return self

    def mode(self, saveMode: str) -> DataFrameWriter:
        self._mode = saveMode
        return self

    def option(self, key: str, value: object) -> DataFrameWriter:
        self._options.append((key, _option_text(value)))
        return self

    def options(self, **options: object) -> DataFrameWriter:
        for key, value in options.items():
            self.option(key, value)
        return self

    def partitionBy(self, *cols: str | list[str] | tuple[str, ...]) -> DataFrameWriter:
        """Names of columns to partition by; each call replaces the columns of
        the one before. Only column names (``str``) are
        accepted; ``Column`` objects raise a type error. The engine writes no
        partitioned layout, so any column fails ``save`` with
        ``KRISHIV_UNSUPPORTED_FEATURE``; an empty ``partitionBy`` is a no-op,
        as in Spark."""
        names: list[str] = []
        for c in cols:
            if isinstance(c, str):
                names.append(c)
            elif isinstance(c, list | tuple):
                for item in c:
                    if not isinstance(item, str):
                        raise not_expected_type("cols", "str", item)
                    names.append(item)
            else:
                raise not_expected_type("cols", "str", c)
        self._partition_by = names
        return self

    def save(
        self,
        path: str | None = None,
        format: str | None = None,
        mode: str | None = None,
        partitionBy: str | list[str] | tuple[str, ...] | None = None,
        **options: object,
    ) -> None:
        if format is not None:
            self.format(format)
        if mode is not None:
            self.mode(mode)
        if partitionBy is not None:
            self.partitionBy(partitionBy)
        # As PySpark's shortcuts do, skip options given as ``None``.
        self.options(**{k: v for k, v in options.items() if v is not None})
        if path is None:
            raise not_expected_type("path", "str", path)
        self._df._df.write(self._format, self._mode, path, self._options, self._partition_by)

    def parquet(
        self,
        path: str,
        mode: str | None = None,
        partitionBy: str | list[str] | tuple[str, ...] | None = None,
        **options: object,
    ) -> None:
        self.save(path, format="parquet", mode=mode, partitionBy=partitionBy, **options)

    def csv(
        self,
        path: str,
        mode: str | None = None,
        partitionBy: str | list[str] | tuple[str, ...] | None = None,
        **options: object,
    ) -> None:
        self.save(path, format="csv", mode=mode, partitionBy=partitionBy, **options)

    def json(
        self,
        path: str,
        mode: str | None = None,
        partitionBy: str | list[str] | tuple[str, ...] | None = None,
        **options: object,
    ) -> None:
        self.save(path, format="json", mode=mode, partitionBy=partitionBy, **options)


def _option_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class DataFrameWriterV2:
    """Writes a :class:`DataFrame`'s rows to a table (a subset of
    ``pyspark.sql.DataFrameWriterV2``: ``using``/``option``/``options``/
    ``tableProperty``/``partitionedBy``/``clusterBy`` and the ``create``/
    ``replace``/``createOrReplace``/``append``/``overwrite``/
    ``overwritePartitions`` actions).

    Only collection happens here; the engine (``krishiv-session``'s
    ``DataFrameWriterV2``) runs the request through the same command path
    SQL uses (``CREATE TABLE ... AS SELECT`` and friends), so
    ``writeTo("t").create()`` and ``CREATE TABLE t AS SELECT ...`` run one
    code path. ``create`` fails on an existing table with
    ``TABLE_OR_VIEW_ALREADY_EXISTS``; ``append`` is ``INSERT INTO`` by
    position. File-source tables cannot be replaced, so ``replace`` and
    ``createOrReplace`` fail as ``REPLACE TABLE AS SELECT`` does
    (``UNSUPPORTED_FEATURE.TABLE_OPERATION``); ``overwrite`` and
    ``overwritePartitions`` always fail with ``KRISHIV_UNSUPPORTED_FEATURE``
    naming ``INSERT OVERWRITE``; partitioned or clustered
    tables fail the same way, as ``CREATE TABLE ... PARTITIONED BY`` does.
    """

    def __init__(self, df: DataFrame, table: str) -> None:
        self._df = df
        self._table = table
        self._provider: str | None = None
        self._options: list[tuple[str, str]] = []
        self._properties: list[tuple[str, str]] = []
        self._partitioned_by: list[str] = []
        self._cluster_by: list[str] = []

    def using(self, provider: str) -> DataFrameWriterV2:
        if not isinstance(provider, str):
            raise not_expected_type("provider", "str", provider)
        self._provider = provider
        return self

    def option(self, key: str, value: object) -> DataFrameWriterV2:
        if not isinstance(key, str):
            raise not_expected_type("key", "str", key)
        self._options.append((key, _option_text(value)))
        return self

    def options(self, **options: object) -> DataFrameWriterV2:
        for key, value in options.items():
            self.option(key, value)
        return self

    def tableProperty(self, property: str, value: str) -> DataFrameWriterV2:
        if not isinstance(property, str):
            raise not_expected_type("property", "str", property)
        if not isinstance(value, str):
            raise not_expected_type("value", "str", value)
        self._properties.append((property, value))
        return self

    def partitionedBy(self, *cols: str | list[str] | tuple[str, ...]) -> DataFrameWriterV2:
        """Names of columns to partition by; each call replaces the columns of
        the one before. Only column names (``str``) are accepted. The engine
        runs no partitioned tables yet, so any column fails ``create`` (and
        friends) with ``KRISHIV_UNSUPPORTED_FEATURE``; an empty
        ``partitionedBy`` is a no-op, as in Spark."""
        self._partitioned_by = _column_names("cols", cols)
        return self

    def clusterBy(self, *cols: str | list[str] | tuple[str, ...]) -> DataFrameWriterV2:
        """Names of columns to cluster by; each call replaces the columns of
        the one before. Only column names (``str``) are accepted. The engine
        runs no clustered tables yet, so any column fails ``create`` (and
        friends) with ``KRISHIV_UNSUPPORTED_FEATURE``."""
        self._cluster_by = _column_names("cols", cols)
        return self

    def create(self) -> None:
        self._df._df.write_to_table(
            self._table,
            provider=self._provider,
            options=self._options,
            properties=self._properties,
            partitioned_by=self._partitioned_by,
            cluster_by=self._cluster_by,
            mode="create",
        )

    def replace(self) -> None:
        self._df._df.write_to_table(
            self._table,
            provider=self._provider,
            options=self._options,
            properties=self._properties,
            partitioned_by=self._partitioned_by,
            cluster_by=self._cluster_by,
            mode="replace",
        )

    def createOrReplace(self) -> None:
        self._df._df.write_to_table(
            self._table,
            provider=self._provider,
            options=self._options,
            properties=self._properties,
            partitioned_by=self._partitioned_by,
            cluster_by=self._cluster_by,
            mode="createOrReplace",
        )

    def append(self) -> None:
        self._df._df.write_to_table(
            self._table,
            provider=self._provider,
            options=self._options,
            properties=self._properties,
            partitioned_by=self._partitioned_by,
            cluster_by=self._cluster_by,
            mode="append",
        )

    def overwrite(self, condition: Column | str) -> None:
        """Overwrites the rows ``condition`` selects. This always fails with
        ``KRISHIV_UNSUPPORTED_FEATURE`` naming ``INSERT OVERWRITE``; the condition (a
        :class:`Column` or SQL text) never runs."""
        if not isinstance(condition, Column | str):
            raise not_expected_type("condition", "Column or str", condition)
        self._df._df.write_to_table(
            self._table,
            provider=self._provider,
            options=self._options,
            properties=self._properties,
            partitioned_by=self._partitioned_by,
            cluster_by=self._cluster_by,
            mode="overwrite",
            condition=condition if isinstance(condition, str) else "",
        )

    def overwritePartitions(self) -> None:
        self._df._df.write_to_table(
            self._table,
            provider=self._provider,
            options=self._options,
            properties=self._properties,
            partitioned_by=self._partitioned_by,
            cluster_by=self._cluster_by,
            mode="overwritePartitions",
        )


def _column_names(arg: str, cols: tuple[object, ...]) -> list[str]:
    """Column names from ``f("a", "b")`` or ``f(["a", "b"])``; anything but
    ``str`` raises a type error, as ``DataFrameWriter.partitionBy`` does."""
    names: list[str] = []
    for c in cols:
        if isinstance(c, str):
            names.append(c)
        elif isinstance(c, list | tuple):
            for item in c:
                if not isinstance(item, str):
                    raise not_expected_type(arg, "str", item)
                names.append(item)
        else:
            raise not_expected_type(arg, "str", c)
    return names
