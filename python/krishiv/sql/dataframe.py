# SPDX-License-Identifier: Apache-2.0
"""``DataFrame``: a lazily evaluated query, as ``pyspark.sql.DataFrame`` (transformations and analysis;
the actions arrive with the executor)."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, cast

from krishiv import _native
from krishiv.errors import (
    PySparkAttributeError,
    PySparkTypeError,
    PySparkValueError,
    not_expected_type,
    not_implemented,
)
from krishiv.sql.column import Column, _named
from krishiv.sql.group import GroupedData
from krishiv.sql.types import StructType, _struct_from_columns

if TYPE_CHECKING:
    from krishiv.sql.session import SparkSession

_HOW = {
    "": "inner",
    "inner": "inner",
    "cross": "cross",
    "outer": "full",
    "full": "full",
    "fullouter": "full",
    "full_outer": "full",
    "left": "left",
    "leftouter": "left",
    "left_outer": "left",
    "right": "right",
    "rightouter": "right",
    "right_outer": "right",
    "semi": "semi",
    "leftsemi": "semi",
    "left_semi": "semi",
    "anti": "anti",
    "leftanti": "anti",
    "left_anti": "anti",
}


ColumnOrName = Column | str
ColumnsOrNames = ColumnOrName | list[ColumnOrName] | tuple[ColumnOrName, ...]


def _flatten(cols: tuple[Any, ...]) -> list[Any]:
    """``f("a", "b")`` and ``f(["a", "b"])`` are the same call."""
    if len(cols) == 1 and isinstance(cols[0], list | tuple):
        return list(cols[0])
    return list(cols)


class DataFrame:
    """A query over a session's data. Transformations return a new DataFrame."""

    def __init__(self, native: _native.DataFrame, session: SparkSession) -> None:
        self._df = native
        self._session = session

    @property
    def sparkSession(self) -> SparkSession:
        return self._session

    # Analysis (no execution).
    @property
    def schema(self) -> StructType:
        """The result schema, by analysis."""
        return _struct_from_columns(self._df.schema())

    @property
    def columns(self) -> list[str]:
        return [name for name, _, _ in self._df.schema()]

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        return [(name, type_) for name, type_, _ in self._df.schema()]

    def printSchema(self) -> None:
        print("root")
        for name, type_, nullable in self._df.schema():
            print(f" |-- {name}: {type_} (nullable = {str(nullable).lower()})")

    # Column access.
    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return Column(self._df.col(item))
        if isinstance(item, Column):
            return self.filter(item)
        if isinstance(item, list | tuple):
            return self.select(*item)
        if isinstance(item, int):
            return Column(self._df.col(self.columns[item]))
        raise not_expected_type("item", "int, str, Column, list or tuple", item)

    def __getattr__(self, name: str) -> Column:
        if name.startswith("_") or name not in self.columns:
            raise PySparkAttributeError(f"'DataFrame' object has no attribute '{name}'")
        return Column(self._df.col(name))

    # Transformations.
    def _new(self, native: _native.DataFrame) -> DataFrame:
        return DataFrame(native, self._session)

    def select(self, *cols: ColumnsOrNames) -> DataFrame:
        return self._new(self._df.select([_named(c) for c in _flatten(cols)]))

    def selectExpr(self, *expr: str) -> DataFrame:
        raise not_implemented("selectExpr (SQL expression strings)")

    def filter(self, condition: Column | str) -> DataFrame:
        if isinstance(condition, str):
            raise not_implemented("filter/where with a SQL expression string")
        if not isinstance(condition, Column):
            raise not_expected_type("condition", "Column or str", condition)
        return self._new(self._df.filter(condition._c))

    where = filter

    def withColumn(self, colName: str, col: Column) -> DataFrame:
        if not isinstance(col, Column):
            raise not_expected_type("col", "Column", col)
        return self._new(self._df.with_column(colName, col._c))

    def alias(self, alias: str) -> DataFrame:
        return self._new(self._df.alias(alias))

    def groupBy(self, *cols: ColumnsOrNames) -> GroupedData:
        return GroupedData(self._df.group_by([_named(c) for c in _flatten(cols)]), self)

    groupby = groupBy

    def agg(self, *exprs: Column) -> DataFrame:
        return self.groupBy().agg(*exprs)

    def join(
        self,
        other: DataFrame,
        on: str | list[str] | Column | list[Column] | None = None,
        how: str | None = None,
    ) -> DataFrame:
        if not isinstance(other, DataFrame):
            raise not_expected_type("other", "DataFrame", other)
        kind = _HOW.get((how or "inner").lower())
        if kind is None:
            raise PySparkValueError(
                errorClass="VALUE_NOT_ALLOWED",
                messageParameters={"arg_name": "how", "allowed_values": ", ".join(sorted(_HOW))},
            )
        if on is None:
            if kind not in ("inner", "cross"):
                raise PySparkValueError("a join without a condition must be inner or cross")
            return self._new(self._df.join(other._df, how="cross"))
        items = list(on) if isinstance(on, list | tuple) else [on]
        if all(isinstance(i, str) for i in items):
            return self._new(self._df.join(other._df, using=items, how=kind))
        if all(isinstance(i, Column) for i in items):
            condition: Column = reduce(lambda a, b: a & b, cast("list[Column]", items))
            return self._new(self._df.join(other._df, on=condition._c, how=kind))
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "on",
                "expected_type": "str, Column or a list of one kind",
                "arg_type": "mixed list",
            },
        )

    def crossJoin(self, other: DataFrame) -> DataFrame:
        return self.join(other, how="cross")

    def orderBy(self, *cols: ColumnsOrNames, ascending: bool | list[bool] = True) -> DataFrame:
        keys = _flatten(cols)
        orders = ascending if isinstance(ascending, list) else [ascending] * len(keys)
        if len(orders) != len(keys):
            raise PySparkValueError("ascending must have one value per sort column")
        ordered = [
            _named(k) if asc else _named(k).sort_order(False, False) for k, asc in zip(keys, orders, strict=True)
        ]
        return self._new(self._df.order_by(ordered))

    sort = orderBy

    def limit(self, num: int) -> DataFrame:
        if not isinstance(num, int):
            raise not_expected_type("num", "int", num)
        return self._new(self._df.limit(num))
