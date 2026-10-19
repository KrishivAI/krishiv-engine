# SPDX-License-Identifier: Apache-2.0
"""``GroupedData``: a DataFrame grouped by key columns, as ``pyspark.sql.GroupedData``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from krishiv import _native
from krishiv.errors import not_expected_type, not_implemented
from krishiv.sql.column import Column

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame


class GroupedData:
    def __init__(self, native: _native.GroupedData, df: DataFrame) -> None:
        self._grouped = native
        self._source = df

    def agg(self, *exprs: Column) -> DataFrame:
        from krishiv.sql.dataframe import DataFrame

        if len(exprs) == 1 and isinstance(exprs[0], dict):
            raise not_implemented("agg with a dict")
        for e in exprs:
            if not isinstance(e, Column):
                raise not_expected_type("exprs", "Column", e)
        return DataFrame(self._grouped.agg([e._c for e in exprs]), self._source._session)

    def count(self) -> DataFrame:
        from krishiv.sql import functions as F

        return self.agg(F.count("*").alias("count"))
