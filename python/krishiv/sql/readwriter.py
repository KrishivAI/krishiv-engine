# SPDX-License-Identifier: Apache-2.0
"""``DataFrameReader``: ``spark.read`` (the format and path are checked when the query is analyzed)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from krishiv.errors import not_expected_type
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


def _option_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
