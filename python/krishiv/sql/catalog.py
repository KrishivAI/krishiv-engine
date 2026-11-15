# SPDX-License-Identifier: Apache-2.0
"""The records ``spark.catalog`` returns, as ``pyspark.sql.catalog`` defines them (named tuples)."""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["CatalogMetadata", "Column", "Database", "Function", "Table"]


class CatalogMetadata(NamedTuple):
    """A catalog, as ``pyspark.sql.catalog.CatalogMetadata``."""

    name: str
    description: str | None


class Database(NamedTuple):
    """A database, as ``pyspark.sql.catalog.Database``."""

    name: str
    catalog: str | None
    description: str | None
    locationUri: str


class Table(NamedTuple):
    """A table or view, as ``pyspark.sql.catalog.Table``."""

    name: str
    catalog: str | None
    namespace: list[str] | None
    description: str | None
    tableType: str
    isTemporary: bool


class Column(NamedTuple):
    """A column, as ``pyspark.sql.catalog.Column``."""

    name: str
    description: str | None
    dataType: str
    nullable: bool
    isPartition: bool
    isBucket: bool
    isCluster: bool


class Function(NamedTuple):
    """A function, as ``pyspark.sql.catalog.Function``."""

    name: str
    catalog: str | None
    namespace: list[str] | None
    description: str | None
    className: str
    isTemporary: bool
