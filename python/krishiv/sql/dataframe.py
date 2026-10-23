# SPDX-License-Identifier: Apache-2.0
"""``DataFrame``: a lazily evaluated query, as ``pyspark.sql.DataFrame``.

Transformations and analysis build and check plans; actions (``collect``, ``show``, ``toArrow``, ``toPandas``,
``count``, ``__arrow_c_stream__``) run the query in-process with the GIL released."""

from __future__ import annotations

import importlib
import random
import sys
from collections.abc import Iterable, Iterator
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
from krishiv.sql import _convert
from krishiv.sql._convert import jsonable
from krishiv.sql.column import Column, _literal, _named
from krishiv.sql.functions import first as _first
from krishiv.sql.group import GroupedData
from krishiv.sql.row import Row
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


#: The most rows ``show`` prints, as Spark: ``ByteArrayMethods.MAX_ROUNDED_ARRAY_LENGTH - 1``.
_SHOW_MAX = 2**31 - 17

ColumnOrName = Column | str
ColumnsOrNames = ColumnOrName | list[ColumnOrName] | tuple[ColumnOrName, ...]


def _flatten(cols: tuple[Any, ...]) -> list[Any]:
    """``f("a", "b")`` and ``f(["a", "b"])`` are the same call."""
    if len(cols) == 1 and isinstance(cols[0], list | tuple):
        return list(cols[0])
    return list(cols)


def _quoted(name: str) -> Column:
    """A reference to the output column literally named ``name``: backquoted, so dots and
    backquotes in it are not parsed as a nested path."""
    return Column(_native.col("`" + name.replace("`", "``") + "`"))


def _case_sensitive(session: SparkSession) -> bool:
    """Whether names match case-sensitively, from ``spark.sql.caseSensitive`` (``false`` by default)."""
    return (session.conf.get("spark.sql.caseSensitive", "false") or "false").lower() == "true"


def _same(left: str, right: str, case_sensitive: bool) -> bool:
    """Whether the column name ``left`` matches ``right`` under the session's resolver."""
    return left == right if case_sensitive else left.lower() == right.lower()


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
        if on is not None and kind == "cross":
            kind = "inner"  # Spark: a cross join with a condition is an inner join
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

    def offset(self, num: int) -> DataFrame:
        """Returns a new :class:`DataFrame` by skipping the first ``n`` rows.

        Parameters
        ----------
        num : int
            Number of records to skip.

        Returns
        -------
        :class:`DataFrame`
            Subset of the records

        Examples
        --------
        >>> df = spark.createDataFrame(
        ...     [(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
        >>> df.offset(1).show()
        +---+-----+
        |age| name|
        +---+-----+
        | 23|Alice|
        | 16|  Bob|
        +---+-----+
        >>> df.offset(10).show()
        +---+----+
        |age|name|
        +---+----+
        +---+----+
        """
        if not isinstance(num, int):
            raise not_expected_type("num", "int", num)
        return self._new(self._df.offset(num))

    def distinct(self) -> DataFrame:
        """Returns a new :class:`DataFrame` containing the distinct rows in this :class:`DataFrame`.

        Returns
        -------
        :class:`DataFrame`
            DataFrame with distinct records.

        See Also
        --------
        DataFrame.dropDuplicates : Remove duplicate rows from this DataFrame.

        Examples
        --------
        >>> df = spark.createDataFrame(
        ...     [(14, "Tom"), (23, "Alice"), (23, "Alice")], ["age", "name"])
        >>> df.distinct().sort("age").show()
        +---+-----+
        |age| name|
        +---+-----+
        | 14|  Tom|
        | 23|Alice|
        +---+-----+

        >>> df.distinct().count()
        2
        """
        return self._new(self._df.distinct())

    def dropDuplicates(self, subset: list[str] | tuple[str, ...] | None = None) -> DataFrame:
        """Return a new :class:`DataFrame` with duplicate rows removed,
        optionally only considering certain columns.

        :func:`drop_duplicates` is an alias for :func:`dropDuplicates`.

        Parameters
        ----------
        subset : list of column names, optional
            List of columns to use for duplicate comparison (default All columns).

        Returns
        -------
        :class:`DataFrame`
            DataFrame without duplicates.

        Examples
        --------
        >>> from krishiv.sql import Row
        >>> df = spark.createDataFrame([
        ...     Row(name='Alice', age=5, height=80),
        ...     Row(name='Alice', age=5, height=80),
        ...     Row(name='Alice', age=10, height=80),
        ... ])

        Deduplicate the same rows.

        >>> df.dropDuplicates().sort("age").show()
        +-----+---+------+
        | name|age|height|
        +-----+---+------+
        |Alice|  5|    80|
        |Alice| 10|    80|
        +-----+---+------+

        Deduplicate values on 'name' and 'height' columns.

        >>> df.dropDuplicates(['name', 'height']).sort("age").show()
        +-----+---+------+
        | name|age|height|
        +-----+---+------+
        |Alice|  5|    80|
        +-----+---+------+
        """
        if subset is None:
            return self._new(self._df.distinct())
        if isinstance(subset, str) or not isinstance(subset, Iterable):
            raise not_expected_type("subset", "list or tuple", subset)
        keys = list(subset)
        for c in keys:
            if not isinstance(c, str):
                raise not_expected_type("subset", "str", c)
        # One row per group of equal keys, as Spark's ``Deduplicate`` keeps one row per group;
        # the row kept is arbitrary in both. The engine groups, keeps the first row's values,
        # and restores the column order.
        unique = list(dict.fromkeys(keys))
        case_sensitive = _case_sensitive(self._session)
        others = [c for c in self.columns if not any(_same(c, k, case_sensitive) for k in unique)]
        grouped = self.groupBy([_quoted(k) for k in unique])
        out = grouped.agg(*[_first(_quoted(c)).alias(c) for c in others])
        return out.select([_quoted(c) for c in self.columns])

    drop_duplicates = dropDuplicates

    def drop(self, *cols: ColumnOrName) -> DataFrame:
        """Returns a new :class:`DataFrame` without specified columns.
        This is a no-op if the schema doesn't contain the given column name(s).

        Parameters
        ----------
        cols: str or :class:`Column`
            A name of the column, or the :class:`Column` to be dropped.

        Returns
        -------
        :class:`DataFrame`
            A new :class:`DataFrame` without the specified columns.

        Notes
        -----
        - When an input is a column name, it is treated literally without further interpretation.
          Otherwise, it will try to match the equivalent expression.
          So dropping a column by its name `drop(colName)` has a different semantic
          with directly dropping the column `drop(col(colName))`.

        Examples
        --------
        >>> df = spark.createDataFrame(
        ...     [(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
        >>> df.drop('age').show()
        +-----+
        | name|
        +-----+
        |  Tom|
        |Alice|
        |  Bob|
        +-----+

        >>> df.drop(df.age).show()
        +-----+
        | name|
        +-----+
        |  Tom|
        |Alice|
        |  Bob|
        +-----+
        """
        names: list[str] = []
        exprs: list[Column] = []
        for c in cols:
            if isinstance(c, str):
                names.append(c)
            elif isinstance(c, Column):
                exprs.append(c)
            else:
                raise not_expected_type("col", "Column or str", c)
        case_sensitive = _case_sensitive(self._session)
        doomed = {c for c in self.columns if any(_same(c, n, case_sensitive) for n in names)}
        for e in exprs:
            try:
                hits = self.select(e).columns
            except Exception as err:
                get_condition = getattr(err, "getCondition", None)
                condition = get_condition() if callable(get_condition) else None
                if isinstance(condition, str) and condition.startswith("UNRESOLVED"):
                    continue  # a column that resolves to nothing drops nothing, as Spark
                raise
            # Matched by output name: an aliased expression resolving to a same-named column
            # drops it, where Spark's expression-identity check would not.
            doomed.update(h for h in hits if h in self.columns)
        if not doomed:
            return self._new(self._df)
        return self.select([_quoted(c) for c in self.columns if c not in doomed])

    def withColumnRenamed(self, existing: str, new: str) -> DataFrame:
        """Returns a new :class:`DataFrame` by renaming an existing column.
        This is a no-op if the schema doesn't contain the given column name.

        Parameters
        ----------
        existing : str
            The name of the existing column to be renamed.
        new : str
            The new name to be assigned to the column.

        Returns
        -------
        :class:`DataFrame`
            A new DataFrame with renamed column.

        See Also
        --------
        DataFrame.withColumnsRenamed

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], schema=["age", "name"])

        Rename a single column

        >>> df.withColumnRenamed("age", "age2").show()
        +----+-----+
        |age2| name|
        +----+-----+
        |   2|Alice|
        |   5|  Bob|
        +----+-----+

        Rename a column that does not exist (no-op)

        >>> df.withColumnRenamed("non_existing", "new_name").show()
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        |  5|  Bob|
        +---+-----+
        """
        return self.withColumnsRenamed({existing: new})

    def withColumnsRenamed(self, colsMap: dict[str, str]) -> DataFrame:
        """Returns a new :class:`DataFrame` by renaming multiple columns.
        This is a no-op if the schema doesn't contain the given column names.

        Parameters
        ----------
        colsMap : dict
            A dict of existing column names and corresponding desired column names.
            Currently, only a single map is supported.

        Returns
        -------
        :class:`DataFrame`
            DataFrame with renamed columns.

        See Also
        --------
        DataFrame.withColumnRenamed

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], schema=["age", "name"])

        Rename a single column

        >>> df.withColumnsRenamed({"age": "age2"}).show()
        +----+-----+
        |age2| name|
        +----+-----+
        |   2|Alice|
        |   5|  Bob|
        +----+-----+

        Rename multiple columns

        >>> df.withColumnsRenamed({"age": "age2", "name": "name2"}).show()
        +----+-----+
        |age2|name2|
        +----+-----+
        |   2|Alice|
        |   5|  Bob|
        +----+-----+

        Rename non-existing column (no-op)

        >>> df.withColumnsRenamed({"non_existing": "new_name"}).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        |  5|  Bob|
        +---+-----+

        Rename with an empty dictionary (no-op)

        >>> df.withColumnsRenamed({}).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        |  5|  Bob|
        +---+-----+
        """
        if not isinstance(colsMap, dict):
            raise not_expected_type("colsMap", "dict", colsMap)
        for k, v in colsMap.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise not_expected_type("colsMap", "dict[str, str]", colsMap)
        case_sensitive = _case_sensitive(self._session)
        if not colsMap or not any(_same(c, k, case_sensitive) for c in self.columns for k in colsMap):
            return self._new(self._df)

        def renamed(c: str) -> str:
            for k, v in colsMap.items():
                if _same(c, k, case_sensitive):
                    return v
            return c

        return self.select([_quoted(c).alias(renamed(c)) for c in self.columns])

    def union(self, other: DataFrame) -> DataFrame:
        """Return a new :class:`DataFrame` containing the union of rows in this and another
        :class:`DataFrame`.

        Parameters
        ----------
        other : :class:`DataFrame`
            Another :class:`DataFrame` that needs to be unioned.

        Returns
        -------
        :class:`DataFrame`
            A new :class:`DataFrame` containing the combined rows with corresponding columns.

        See Also
        --------
        DataFrame.unionAll

        Notes
        -----
        This method performs a SQL-style set union of the rows from both `DataFrame` objects,
        with no automatic deduplication of elements.

        Use the `distinct()` method to perform deduplication of rows.

        The method resolves columns by position (not by name), following the standard behavior
        in SQL.

        Examples
        --------
        >>> df1 = spark.createDataFrame([(1, 'A'), (2, 'B')], ['id', 'value'])
        >>> df2 = spark.createDataFrame([(3, 'C'), (4, 'D')], ['id', 'value'])
        >>> df3 = df1.union(df2)
        >>> df3.sort("id").show()
        +---+-----+
        | id|value|
        +---+-----+
        |  1|    A|
        |  2|    B|
        |  3|    C|
        |  4|    D|
        +---+-----+
        """
        if not isinstance(other, DataFrame):
            raise not_expected_type("other", "DataFrame", other)
        return self._new(self._df.union(other._df))

    def unionAll(self, other: DataFrame) -> DataFrame:
        """:func:`unionAll` is an alias for :func:`union`."""
        return self.union(other)

    def unionByName(self, other: DataFrame, allowMissingColumns: bool = False) -> DataFrame:
        """Returns a new :class:`DataFrame` containing union of rows in this and another
        :class:`DataFrame`.

        This method performs a union operation on both input DataFrames, resolving columns by
        name (rather than position). When `allowMissingColumns` is True, missing columns will
        be filled with null.

        Parameters
        ----------
        other : :class:`DataFrame`
            Another :class:`DataFrame` that needs to be combined.
        allowMissingColumns : bool, optional, default False
           Specify whether to allow missing columns.

        Returns
        -------
        :class:`DataFrame`
            A new :class:`DataFrame` containing the combined rows with corresponding
            columns of the two given DataFrames.

        Examples
        --------
        >>> df1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
        >>> df2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col0"])
        >>> df1.unionByName(df2).show()
        +----+----+----+
        |col0|col1|col2|
        +----+----+----+
        |   1|   2|   3|
        |   6|   4|   5|
        +----+----+----+

        >>> df1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
        >>> df2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col3"])
        >>> df1.unionByName(df2, allowMissingColumns=True).show()
        +----+----+----+----+
        |col0|col1|col2|col3|
        +----+----+----+----+
        |   1|   2|   3|NULL|
        |NULL|   4|   5|   6|
        +----+----+----+----+
        """
        if not isinstance(other, DataFrame):
            raise not_expected_type("other", "DataFrame", other)
        return self._new(self._df.union_by_name(other._df, allowMissingColumns))

    def intersect(self, other: DataFrame) -> DataFrame:
        """Return a new :class:`DataFrame` containing rows only in
        both this :class:`DataFrame` and another :class:`DataFrame`.
        Note that any duplicates are removed. To preserve duplicates
        use :func:`intersectAll`.

        Parameters
        ----------
        other : :class:`DataFrame`
            Another :class:`DataFrame` that needs to be combined.

        Returns
        -------
        :class:`DataFrame`
            Combined DataFrame.

        Notes
        -----
        This is equivalent to `INTERSECT` in SQL.

        Examples
        --------
        >>> df1 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3), ("c", 4)], ["C1", "C2"])
        >>> df2 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3)], ["C1", "C2"])
        >>> df1.intersect(df2).sort("C1", "C2").show()
        +---+---+
        | C1| C2|
        +---+---+
        |  a|  1|
        |  b|  3|
        +---+---+
        """
        if not isinstance(other, DataFrame):
            raise not_expected_type("other", "DataFrame", other)
        return self._new(self._df.intersect(other._df))

    def intersectAll(self, other: DataFrame) -> DataFrame:
        """Return a new :class:`DataFrame` containing rows in both this :class:`DataFrame`
        and another :class:`DataFrame` while preserving duplicates.

        This is equivalent to `INTERSECT ALL` in SQL. As standard in SQL, this function
        resolves columns by position (not by name).

        Returns
        -------
        :class:`DataFrame`
            Combined DataFrame.

        Examples
        --------
        >>> df1 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3), ("c", 4)], ["C1", "C2"])
        >>> df2 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3)], ["C1", "C2"])
        >>> df1.intersectAll(df2).sort("C1", "C2").show()
        +---+---+
        | C1| C2|
        +---+---+
        |  a|  1|
        |  a|  1|
        |  b|  3|
        +---+---+
        """
        if not isinstance(other, DataFrame):
            raise not_expected_type("other", "DataFrame", other)
        return self._new(self._df.intersect_all(other._df))

    def exceptAll(self, other: DataFrame) -> DataFrame:
        """Return a new :class:`DataFrame` containing rows in this :class:`DataFrame` but
        not in another :class:`DataFrame` while preserving duplicates.

        This is equivalent to `EXCEPT ALL` in SQL.
        As standard in SQL, this function resolves columns by position (not by name).

        Parameters
        ----------
        other : :class:`DataFrame`
            The other :class:`DataFrame` to compare to.

        Returns
        -------
        :class:`DataFrame`

        See Also
        --------
        DataFrame.subtract : Similar to `exceptAll`, but eliminates duplicates.

        Examples
        --------
        >>> df1 = spark.createDataFrame(
        ...         [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
        >>> df2 = spark.createDataFrame([("a", 1), ("b", 3)], ["C1", "C2"])
        >>> df1.exceptAll(df2).sort("C1", "C2").show()
        +---+---+
        | C1| C2|
        +---+---+
        |  a|  1|
        |  a|  1|
        |  a|  2|
        |  c|  4|
        +---+---+
        """
        if not isinstance(other, DataFrame):
            raise not_expected_type("other", "DataFrame", other)
        return self._new(self._df.except_all(other._df))

    def subtract(self, other: DataFrame) -> DataFrame:
        """Return a new :class:`DataFrame` containing rows in this :class:`DataFrame`
        but not in another :class:`DataFrame`.

        Parameters
        ----------
        other : :class:`DataFrame`
            Another :class:`DataFrame` that needs to be subtracted.

        Returns
        -------
        :class:`DataFrame`
            Subtracted DataFrame.

        Notes
        -----
        This is equivalent to `EXCEPT DISTINCT` in SQL.

        See Also
        --------
        DataFrame.exceptAll : Similar to `subtract`, but preserves duplicates.

        Examples
        --------
        >>> df1 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3), ("c", 4)], ["C1", "C2"])
        >>> df2 = spark.createDataFrame([("a", 1), ("a", 1), ("b", 3)], ["C1", "C2"])
        >>> df1.subtract(df2).sort("C1", "C2").show()
        +---+---+
        | C1| C2|
        +---+---+
        |  c|  4|
        +---+---+
        """
        if not isinstance(other, DataFrame):
            raise not_expected_type("other", "DataFrame", other)
        return self._new(self._df.subtract(other._df))

    def sample(
        self,
        withReplacement: bool | float | None = None,
        fraction: float | None = None,
        seed: int | None = None,
    ) -> DataFrame:
        """Returns a sampled subset of this :class:`DataFrame`.

        .. note:: The engine cannot sample yet: the returned :class:`DataFrame` fails its
            first action with ``KRISHIV_UNSUPPORTED_FEATURE``.

        Parameters
        ----------
        withReplacement : bool, optional
            Sample with replacement or not (default ``False``).
        fraction : float, optional
            Fraction of rows to generate, range [0.0, 1.0].
        seed : int, optional
            Seed for sampling (default a random seed).

        Returns
        -------
        :class:`DataFrame`
            Sampled rows from given DataFrame.

        Notes
        -----
        This is not guaranteed to provide exactly the fraction specified of the total
        count of the given :class:`DataFrame`.

        `fraction` is required and, `withReplacement` and `seed` are optional.
        """
        if isinstance(withReplacement, bool) and isinstance(fraction, float):
            replace, frac, sampled_seed = withReplacement, fraction, seed
        elif withReplacement is None and isinstance(fraction, float):
            replace, frac, sampled_seed = False, fraction, seed
        elif isinstance(withReplacement, float):
            replace, frac, sampled_seed = False, float(withReplacement), fraction
        else:
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "expected_type": "bool, float or int",
                    "arg_name": "withReplacement (optional), fraction (required) and seed (optional)",
                    "arg_type": ", ".join(type(a).__name__ for a in (withReplacement, fraction, seed)),
                },
            )
        sampled = int(sampled_seed) if sampled_seed is not None else random.randint(0, sys.maxsize)
        return self._new(self._df.sample(replace, frac, sampled))

    def repartition(self, numPartitions: int | ColumnOrName, *cols: ColumnOrName) -> DataFrame:
        """Returns a new :class:`DataFrame` partitioned by the given partitioning expressions.

        .. note:: Embedded Krishiv runs on one node: the partitioning is accepted and ignored,
            and the rows never change. ``spark_partition_id`` is not available.

        Parameters
        ----------
        numPartitions : int
            can be an int to specify the target number of partitions or a Column.
            If it is a Column, it will be used as the first partitioning column. If not specified,
            the default number of partitions is used.
        cols : str or :class:`Column`
            partitioning columns.

        Returns
        -------
        :class:`DataFrame`
            Repartitioned DataFrame.
        """
        if isinstance(numPartitions, int):
            targets: tuple[ColumnOrName, ...] = cols
        elif isinstance(numPartitions, str | Column):
            targets = (numPartitions, *cols)
        else:
            raise not_expected_type("numPartitions", "Column or str", numPartitions)
        for c in targets:
            if not isinstance(c, str | Column):
                raise not_expected_type("cols", "Column or str", c)
        return self._new(self._df.repartition())

    def coalesce(self, numPartitions: int) -> DataFrame:
        """Returns a new :class:`DataFrame` with exactly ``numPartitions`` partitions.

        .. note:: Embedded Krishiv runs on one node: the partition count is accepted and
            ignored, and the rows never change.

        Parameters
        ----------
        numPartitions : int
            specify the target number of partitions

        Returns
        -------
        :class:`DataFrame`
        """
        if not isinstance(numPartitions, int):
            raise not_expected_type("numPartitions", "int", numPartitions)
        return self._new(self._df.coalesce())

    def hint(self, name: str, *parameters: Any) -> DataFrame:
        """Specifies some hint on the current :class:`DataFrame`.

        Parameters
        ----------
        name : str
            A name of the hint.
        parameters : str, list, float or int
            Optional parameters.

        Returns
        -------
        :class:`DataFrame`
            Hinted DataFrame

        Notes
        -----
        A hint the engine cannot follow (an unknown hint, a hint naming no relation, or a
        partitioning hint, which has no effect on one node) comes back as a warning, as Spark
        logs it; the rows never change. Use ``explain`` to see the recorded hint.
        """
        if len(parameters) == 1 and isinstance(parameters[0], list):
            parameters = parameters[0]
        if not isinstance(name, str):
            raise not_expected_type("name", "str", name)
        allowed = (str, float, int, Column, list)
        primitives = (str, float, int)
        allowed_repr = ", ".join([t.__name__ for t in allowed[:-1]] + [f"list[{t.__name__}]" for t in primitives])
        converted: list[Any] = []
        for p in parameters:
            if not isinstance(p, allowed):
                raise PySparkTypeError(
                    errorClass="DISALLOWED_TYPE_FOR_CONTAINER",
                    messageParameters={
                        "arg_name": "parameters",
                        "arg_type": type(parameters).__name__,
                        "allowed_types": allowed_repr,
                        "item_type": type(p).__name__,
                    },
                )
            if isinstance(p, list):
                if not all(isinstance(e, primitives) for e in p):
                    raise PySparkTypeError(
                        errorClass="DISALLOWED_TYPE_FOR_CONTAINER",
                        messageParameters={
                            "arg_name": "parameters",
                            "arg_type": type(parameters).__name__,
                            "allowed_types": allowed_repr,
                            "item_type": type(p).__name__ + "[" + type(p[0]).__name__ + "]",
                        },
                    )
                # The engine has no array literal: each element becomes its own parameter.
                converted.extend(_literal(e) for e in p)
            elif isinstance(p, Column):
                converted.append(p._c)
            else:
                converted.append(_literal(p))
        return self._new(self._df.hint(name, converted))

    def unpivot(
        self,
        ids: ColumnOrName | list[ColumnOrName] | tuple[ColumnOrName, ...],
        values: ColumnOrName | list[ColumnOrName] | tuple[ColumnOrName, ...] | None,
        variableColumnName: str,
        valueColumnName: str,
    ) -> DataFrame:
        """Unpivot a DataFrame from wide format to long format, optionally leaving
        identifier columns set.

        Parameters
        ----------
        ids : str, Column, tuple, list
            Column(s) to use as identifiers. Can be a single column or column name,
            or a list or tuple for multiple columns.
        values : str, Column, tuple, list, optional
            Column(s) to unpivot. Can be a single column or column name, or a list or tuple
            for multiple columns. If specified, must not be empty. If not specified, uses all
            columns that are not set as `ids`.
        variableColumnName : str
            Name of the variable column.
        valueColumnName : str
            Name of the value column.

        Returns
        -------
        :class:`DataFrame`
            Unpivoted DataFrame.

        See Also
        --------
        DataFrame.melt

        Examples
        --------
        >>> df = spark.createDataFrame(
        ...     [(1, 11, 1.1), (2, 12, 1.2)],
        ...     ["id", "int", "double"],
        ... )
        >>> df.unpivot("id", ["int", "double"], "var", "val").sort("id", "var").show()
        +---+------+----+
        | id|   var| val|
        +---+------+----+
        |  1|double| 1.1|
        |  1|   int|11.0|
        |  2|double| 1.2|
        |  2|   int|12.0|
        +---+------+----+
        """
        assert ids is not None, "ids must not be None"
        if not isinstance(variableColumnName, str):
            raise not_expected_type("variableColumnName", "str", variableColumnName)
        if not isinstance(valueColumnName, str):
            raise not_expected_type("valueColumnName", "str", valueColumnName)
        id_list = ids if isinstance(ids, list | tuple) else [ids]
        for c in id_list:
            if not isinstance(c, str | Column):
                raise not_expected_type("ids", "Column or str", c)
        id_cols = [_named(c) if isinstance(c, str) else c._c for c in id_list]
        if values is None:
            case_sensitive = _case_sensitive(self._session)
            id_names = {c for c in id_list if isinstance(c, str)}
            for c in id_list:
                if not isinstance(c, str):
                    id_names.update(n for n in self.select(c).columns if n in self.columns)
            pairs = [(c, _quoted(c)._c) for c in self.columns if not any(_same(c, n, case_sensitive) for n in id_names)]
        else:
            value_list = values if isinstance(values, list | tuple) else [values]
            pairs = []
            for c in value_list:
                if isinstance(c, str):
                    pairs.append((c, _named(c)))
                elif isinstance(c, Column):
                    (label,) = self.select(c).columns
                    pairs.append((label, c._c))
                else:
                    raise not_expected_type("values", "Column or str", c)
        return self._new(self._df.unpivot(id_cols, pairs, variableColumnName, valueColumnName))

    def melt(
        self,
        ids: ColumnOrName | list[ColumnOrName] | tuple[ColumnOrName, ...],
        values: ColumnOrName | list[ColumnOrName] | tuple[ColumnOrName, ...] | None,
        variableColumnName: str,
        valueColumnName: str,
    ) -> DataFrame:
        """:func:`melt` is an alias for :func:`unpivot`.

        See Also
        --------
        DataFrame.unpivot
        """
        return self.unpivot(ids, values, variableColumnName, valueColumnName)

    def transform(self, func: Any, *args: Any, **kwargs: Any) -> DataFrame:
        """Returns a new :class:`DataFrame`. Concise syntax for chaining custom transformations.

        Parameters
        ----------
        func : function
            a function that takes and returns a :class:`DataFrame`.
        *args
            Positional arguments to pass to func.
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        :class:`DataFrame`
            Transformed DataFrame.

        Examples
        --------
        >>> from krishiv.sql import functions as F
        >>> df = spark.createDataFrame([(1, 1.0), (2, 2.0)], ["int", "float"])
        >>> def cast_all_to_int(input_df):
        ...     return input_df.select([F.col(c).cast("int") for c in input_df.columns])
        ...
        >>> df.transform(cast_all_to_int).sort("int").show()
        +---+-----+
        |int|float|
        +---+-----+
        |  1|    1|
        |  2|    2|
        +---+-----+
        """
        result = func(self, *args, **kwargs)
        assert isinstance(result, DataFrame), (
            f"Func returned an instance of type [{type(result)}], should have been DataFrame."
        )
        return result

    def toDF(self, *cols: str) -> DataFrame:
        """Returns a new :class:`DataFrame` with new specified column names

        Parameters
        ----------
        *cols : tuple
            a tuple of string new column name. The length of the
            list needs to be the same as the number of columns in the initial
            :class:`DataFrame`

        Returns
        -------
        :class:`DataFrame`
            DataFrame with new column names.

        Examples
        --------
        >>> df = spark.createDataFrame([(14, "Tom"), (23, "Alice"),
        ...     (16, "Bob")], ["age", "name"])
        >>> df.toDF('f1', 'f2').sort("f1").show()
        +---+-----+
        | f1|   f2|
        +---+-----+
        | 14|  Tom|
        | 23|Alice|
        | 16|  Bob|
        +---+-----+
        """
        for c in cols:
            if not isinstance(c, str):
                raise not_expected_type("cols", "list[str]", c)
        return self._new(self._df.to_df(list(cols)))

    def toLocalIterator(self, prefetchPartitions: bool = False) -> Iterator[Row]:
        """Returns an iterator that contains all of the rows in this :class:`DataFrame`.

        .. note:: Embedded Krishiv runs on one node: ``prefetchPartitions`` is accepted and
            ignored.

        Parameters
        ----------
        prefetchPartitions : bool, optional
            If Spark should pre-fetch the next partition before it is needed.

        Returns
        -------
        Iterator
            Iterator of rows.

        Examples
        --------
        >>> df = spark.createDataFrame(
        ...     [(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
        >>> list(df.toLocalIterator())
        [Row(age=14, name='Tom'), Row(age=23, name='Alice'), Row(age=16, name='Bob')]
        """
        return iter(self.collect())

    def toJSON(self, use_unicode: bool = True) -> list[str]:
        """Converts a :class:`DataFrame` into a list of JSON strings, one per row.

        .. note:: Embedded Krishiv has no ``RDD``: this returns a ``list[str]`` where PySpark
            returns an ``RDD[str]``.

        Parameters
        ----------
        use_unicode : bool, optional, default True
            Whether to convert to unicode or not.

        Returns
        -------
        list
            A list of JSON strings, each row as one JSON document.

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], schema=["age", "name"])
        >>> df.sort("age").toJSON()
        ['{"age": 2, "name": "Alice"}', '{"age": 5, "name": "Bob"}']
        """
        import json

        table = self.toArrow()
        zone = self._session_zone()
        names = table.schema.names
        arrow_types = [table.field(name).type for name in names]
        rows = table.to_pylist()
        return [
            json.dumps(
                {n: jsonable(r[n], t, zone) for n, t in zip(names, arrow_types, strict=True)},
                ensure_ascii=not use_unicode,
            )
            for r in rows
        ]

    def tail(self, num: int) -> list[Row]:
        """Returns the last ``num`` rows as a :class:`list` of :class:`Row`.

        Parameters
        ----------
        num : int
            Number of records to return. Will return this number of records
            or all records if the DataFrame contains less than this number of records.

        Returns
        -------
        list
            List of rows

        Examples
        --------
        >>> df = spark.createDataFrame(
        ...     [(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])

        >>> df.tail(2)
        [Row(age=23, name='Alice'), Row(age=16, name='Bob')]
        """
        if not isinstance(num, int):
            raise not_expected_type("num", "int", num)
        rows = self.collect()
        return rows[max(0, len(rows) - num) :]

    # Actions: each runs the query again (as each PySpark action does), with the GIL released.
    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        """The result as an Arrow C stream capsule (the Arrow PyCapsule interface), so pyarrow, Polars and
        DuckDB read it directly. The query runs now; its batches are computed as the consumer reads them,
        and releasing the stream cancels it."""
        return self._df.execute().__arrow_c_stream__(requested_schema)

    def _session_zone(self) -> Any:
        return _convert.session_zone(self._session.conf.get("spark.sql.session.timeZone", "UTC") or "UTC")

    def toArrow(self) -> Any:
        """The result as a ``pyarrow.Table`` (needs pyarrow)."""
        pa = _convert.require_pyarrow("toArrow")
        stream = self._df.execute()
        try:
            table = pa.table(stream)
        except Exception:
            # The C stream carries only an Arrow error message: raise the engine's own error if that was the cause.
            stream.check_error()
            raise
        stream.check_error()
        return table

    def toPandas(self) -> Any:
        """The result as a ``pandas.DataFrame``, converted from Arrow (needs pyarrow and pandas). Instants are
        wall-clock times in the session time zone without a zone, as Spark's ``toPandas``."""
        table = self.toArrow()
        try:
            importlib.import_module("pandas")  # pandas itself is optional: only checked here
        except ImportError as e:
            raise ImportError("toPandas needs pandas; install it with `pip install krishiv[pandas]`") from e
        return _convert.session_local(table, self._session_zone()).to_pandas()

    def collect(self) -> list[Row]:
        """All result rows (needs pyarrow)."""
        return _convert.rows_of(self.toArrow())

    def take(self, num: int) -> list[Row]:
        return self.limit(num).collect()

    def head(self, n: int | None = None) -> Row | list[Row] | None:
        if n is None:
            rows = self.take(1)
            return rows[0] if rows else None
        return self.take(n)

    def first(self) -> Row | None:
        return cast("Row | None", self.head())

    def count(self) -> int:
        """The number of rows."""
        return self._df.count()

    def show(self, n: int = 20, truncate: bool | int = True, vertical: bool = False) -> None:
        """Prints the first ``n`` rows as a table (needs pyarrow)."""
        if not isinstance(n, int) or isinstance(n, bool):
            raise not_expected_type("n", "int", n)
        if not isinstance(truncate, bool | int):
            raise not_expected_type("truncate", "bool or int", truncate)
        rows = min(max(n, 0), _SHOW_MAX)  # Spark's bounds: MAX_ROUNDED_ARRAY_LENGTH - 1
        cells = _convert.table_cells(self.limit(rows + 1).toArrow(), self._session_zone())
        print(_convert.show_string(self.columns, cells, rows, truncate, vertical), end="")

    def explain(
        self,
        extended: bool | str | None = None,
        mode: str | None = None,
        *,
        analyze: bool = False,
        verbose: bool = False,
    ) -> None:
        """Prints the query's plans, as PySpark's ``explain`` (``extended`` or ``mode``: ``simple``,
        ``extended``, ``codegen``, ``cost``, ``formatted``).

        Krishiv extensions (keyword only): ``analyze=True`` runs the query and prints the physical
        plan with each operator's metrics (``EXPLAIN ANALYZE``); ``verbose=True`` prints the logical
        plan after each optimizer rule that changed it (``EXPLAIN VERBOSE``). Use
        :meth:`explainString` for the text or the JSON document instead of printing.
        """
        print(self.explainString(extended, mode, analyze=analyze, verbose=verbose))

    def explainString(
        self,
        extended: bool | str | None = None,
        mode: str | None = None,
        *,
        analyze: bool = False,
        verbose: bool = False,
        json: bool = False,
    ) -> str:
        """What :meth:`explain` prints; with ``json=True`` (``analyze`` or ``verbose`` only) the
        versioned JSON document (``krishiv.query-profile`` or ``krishiv.explain-verbose``;
        experimental: may change before 1.0)."""
        if extended is not None and mode is not None:
            raise PySparkValueError(
                errorClass="CANNOT_SET_TOGETHER", messageParameters={"arg_list": "extended and mode"}
            )
        for name, flag in (("analyze", analyze), ("verbose", verbose), ("json", json)):
            if not isinstance(flag, bool):
                raise not_expected_type(name, "bool", flag)
        if extended is not None and not isinstance(extended, bool | str):
            raise not_expected_type("extended", "bool or str", extended)
        if mode is not None and not isinstance(mode, str):
            raise not_expected_type("mode", "str", mode)
        chosen = mode if mode is not None else extended if isinstance(extended, str) else None
        if analyze or verbose:
            if analyze and verbose:
                raise PySparkValueError(
                    errorClass="CANNOT_SET_TOGETHER", messageParameters={"arg_list": "analyze and verbose"}
                )
            if extended is not None or mode is not None:
                raise PySparkValueError(
                    errorClass="CANNOT_SET_TOGETHER",
                    messageParameters={"arg_list": "analyze or verbose and extended or mode"},
                )
            return self._df.explain("analyze" if analyze else "verbose", json)
        if json:
            raise PySparkValueError(
                errorClass="CANNOT_SET_TOGETHER", messageParameters={"arg_list": "json and a Spark explain mode"}
            )
        if chosen is None:
            chosen = "extended" if extended is True else "simple"
        chosen = chosen.lower()
        # Only Spark's modes here: Krishiv's run through the keywords above.
        if chosen not in ("simple", "extended", "codegen", "cost", "formatted"):
            raise PySparkValueError(f"Unknown explain mode: {chosen}")
        return self._df.explain(chosen, False)

    def createOrReplaceTempView(self, name: str) -> None:
        """Registers this DataFrame as the temporary view ``name`` (views of reads by path only, so far)."""
        if not isinstance(name, str):
            raise not_expected_type("name", "str", name)
        self._df.create_or_replace_temp_view(name)
