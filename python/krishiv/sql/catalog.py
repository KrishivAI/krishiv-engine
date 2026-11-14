# SPDX-License-Identifier: Apache-2.0
"""The result types of ``spark.catalog``, as ``pyspark.sql.catalog`` (named tuples)."""

from __future__ import annotations

from typing import NamedTuple


class CatalogMetadata(NamedTuple):
    """A catalog: its name and description."""

    name: str
    description: str | None


class Database(NamedTuple):
    """A database (namespace)."""

    name: str
    catalog: str | None
    description: str | None
    locationUri: str


class Table(NamedTuple):
    """A table or view."""

    name: str
    catalog: str | None
    namespace: list[str] | None
    description: str | None
    tableType: str
    isTemporary: bool

    @property
    def database(self) -> str | None:
        """The database, when the namespace is a single name."""
        if self.namespace is not None and len(self.namespace) == 1:
            return self.namespace[0]
        return None


class Column(NamedTuple):
    """A column of a table."""

    name: str
    description: str | None
    dataType: str
    nullable: bool
    isPartition: bool
    isBucket: bool
    isCluster: bool


class Function(NamedTuple):
    """A function."""

    name: str
    catalog: str | None
    namespace: list[str] | None
    description: str | None
    className: str
    isTemporary: bool


class TablePartition(NamedTuple):
    """A partition of a table."""

    partition: str
