# SPDX-License-Identifier: Apache-2.0
"""``GroupedData``: a DataFrame grouped by key columns, as ``pyspark.sql.GroupedData``.

Aggregations reuse the group-by aggregation the engine already runs; the
shortcuts and the dict form of ``agg`` build the same expressions the SQL
equivalents build. ``pivot`` builds the plan node SQL's ``PIVOT`` lowers to.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from krishiv import _native
from krishiv.errors import not_expected_type
from krishiv.sql import functions as F
from krishiv.sql.column import Column

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame

#: Engine simple types that ``sum``/``avg``/``min``/``max`` aggregate, as Spark:
#: the integral and floating-point types and decimals.
_NUMERIC = ("tinyint", "smallint", "int", "bigint", "float", "double")


def _numeric(names: list[str], dtypes: list[tuple[str, str]], strict: bool) -> list[str]:
    """The ``names`` that hold numeric values: unknown names pass through
    (analysis reports them, as Spark does). Known non-numeric ones are
    skipped when every column is taken (``strict`` off), and named
    explicitly they fail with ``NOT_NUMERIC_COLUMNS``, as in PySpark."""
    by_name = dict(dtypes)
    out = []
    invalid = []
    for n in names:
        t = by_name.get(n)
        if t is None or t in _NUMERIC or t.startswith("decimal"):
            out.append(n)
        else:
            invalid.append(n)
    if strict and invalid:
        from krishiv.errors import PySparkTypeError

        raise PySparkTypeError(errorClass="NOT_NUMERIC_COLUMNS", messageParameters={"invalid_columns": str(invalid)})
    return out


class GroupedData:
    def __init__(self, native: _native.GroupedData, df: DataFrame) -> None:
        self._grouped = native
        self._source = df
        self._pivoted = False

    def agg(self, *exprs: Column | dict[str, str]) -> DataFrame:
        """Computes aggregates and returns the result as a :class:`DataFrame`.

        Either a single dict from column name to aggregate function name
        (``{"age": "max", "*": "count"}``), or aggregate :class:`Column`
        expressions.

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (3, "Alice"), (5, "Bob")], ["age", "name"])
        >>> df.groupBy("name").agg({"*": "count"}).sort("name").show()
        +-----+--------+
        | name|count(1)|
        +-----+--------+
        |Alice|       2|
        |  Bob|       1|
        +-----+--------+

        >>> df.groupBy("name").agg(F.min(df.age)).sort("name").show()
        +-----+--------+
        | name|min(age)|
        +-----+--------+
        |Alice|       2|
        |  Bob|       5|
        +-----+--------+
        """
        result = self._agg(*exprs)
        if self._pivoted:
            self._analyze_now(result)
        return result

    def _analyze_now(self, result: DataFrame) -> None:
        """Spark analyses a pivot when ``agg`` is called, so its errors come from there; the message of the
        non-aggregate error names the column as written (the engine's relation name is not Spark's)."""
        from krishiv.errors import PySparkException

        try:
            _ = result.schema
        except PySparkException as error:
            if error.getCondition() == "_LEGACY_ERROR_TEMP_1006":
                text = re.sub(r"'__krishiv_\w+?\.", "'", error.getMessage())
                raise type(error)(
                    text,
                    errorClass=error.getCondition(),
                    messageParameters=error.getMessageParameters(),
                    sqlState=error.getSqlState(),
                ) from None
            raise

    def _agg(self, *exprs: Column | dict[str, str]) -> DataFrame:
        from krishiv.sql.dataframe import DataFrame

        if not exprs:
            raise ValueError("exprs should not be empty")
        if len(exprs) == 1 and isinstance(exprs[0], dict):
            columns: list[Column] = []
            for key, func in exprs[0].items():
                if not isinstance(key, str):
                    raise not_expected_type("key", "str", key)
                if not isinstance(func, str):
                    raise not_expected_type("value", "str", func)
                if key == "*" and func == "count":
                    columns.append(F.count("*"))
                else:
                    columns.append(F.call_function(func, F.col(key)))
            return DataFrame(self._grouped.agg([c._c for c in columns]), self._source._session)
        checked: list[Column] = []
        for e in exprs:
            if not isinstance(e, Column):
                raise not_expected_type("exprs", "Column", e)
            checked.append(e)
        return DataFrame(self._grouped.agg([e._c for e in checked]), self._source._session)

    def count(self) -> DataFrame:
        """The number of rows per group, as ``pyspark.sql.GroupedData.count``.

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice"), (3, "Alice"), (5, "Bob")], ["age", "name"])
        >>> df.groupBy("name").count().sort("name").show()
        +-----+-----+
        | name|count|
        +-----+-----+
        |Alice|    2|
        |  Bob|    1|
        +-----+-----+
        """
        return self.agg(F.count("*").alias("count"))

    def _shortcut(self, func: Any, cols: tuple[str, ...]) -> DataFrame:
        for c in cols:
            if not isinstance(c, str):
                raise not_expected_type("cols", "str", c)
        if cols:
            names = _numeric(list(cols), self._source.dtypes, strict=True)
        else:
            # No columns: every numeric column, as Spark aggregates those.
            names = _numeric(self._source.columns, self._source.dtypes, strict=False)
        return self.agg(*[func(F.col(n)) for n in names])

    def sum(self, *cols: str) -> DataFrame:
        """The sum per group of each numeric column, as ``pyspark.sql.GroupedData.sum``.

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice", 80), (3, "Alice", 100), (5, "Bob", 120)],
        ...     ["age", "name", "height"])
        >>> df.groupBy("name").sum("age").sort("name").show()
        +-----+--------+
        | name|sum(age)|
        +-----+--------+
        |Alice|       5|
        |  Bob|       5|
        +-----+--------+
        """
        return self._shortcut(F.sum, cols)

    def avg(self, *cols: str) -> DataFrame:
        """The average per group of each numeric column, as ``pyspark.sql.GroupedData.avg``.

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice", 80), (3, "Alice", 100), (5, "Bob", 120)],
        ...     ["age", "name", "height"])
        >>> df.groupBy("name").avg("age").sort("name").show()
        +-----+--------+
        | name|avg(age)|
        +-----+--------+
        |Alice|     2.5|
        |  Bob|     5.0|
        +-----+--------+
        """
        return self._shortcut(F.avg, cols)

    mean = avg

    def min(self, *cols: str) -> DataFrame:
        """The minimum per group of each numeric column, as ``pyspark.sql.GroupedData.min``.

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice", 80), (3, "Alice", 100), (5, "Bob", 120)],
        ...     ["age", "name", "height"])
        >>> df.groupBy("name").min("age").sort("name").show()
        +-----+--------+
        | name|min(age)|
        +-----+--------+
        |Alice|       2|
        |  Bob|       5|
        +-----+--------+
        """
        return self._shortcut(F.min, cols)

    def max(self, *cols: str) -> DataFrame:
        """The maximum per group of each numeric column, as ``pyspark.sql.GroupedData.max``.

        Examples
        --------
        >>> df = spark.createDataFrame([(2, "Alice", 80), (3, "Alice", 100), (5, "Bob", 120)],
        ...     ["age", "name", "height"])
        >>> df.groupBy("name").max("age").sort("name").show()
        +-----+--------+
        | name|max(age)|
        +-----+--------+
        |Alice|       3|
        |  Bob|       5|
        +-----+--------+
        """
        return self._shortcut(F.max, cols)

    def pivot(self, pivot_col: str, values: list[Any] | None = None) -> GroupedData:
        """Pivots a column of the DataFrame and performs the specified aggregation, as
        ``pyspark.sql.GroupedData.pivot``.

        Parameters
        ----------
        pivot_col : str
            Name of the column to pivot.
        values : list, optional
            List of values that will be translated to columns in the output DataFrame. When omitted (or
            empty) the distinct values are read from the DataFrame first, sorted ascending with NULL first
            (at most ``spark.sql.pivotMaxValues``, default 10000).

        Notes
        -----
        An output column is named by the value (``2012``), or, with several aggregates, by the value and the
        aggregate (``2012_sum(x)``, ``2012_s`` for an alias). A group with no row for a value has NULL there
        (``count`` too), except for aggregates such as ``collect_list``, which give an empty array.

        Examples
        --------
        >>> df = spark.createDataFrame([(2012, "Java", 20000), (2012, "dotNET", 5000), (2013, "Java", 30000)],
        ...     ["year", "course", "earnings"])
        >>> df.groupBy("year").pivot("course", ["Java", "dotNET"]).sum("earnings").sort("year").show()
        +----+-----+------+
        |year| Java|dotNET|
        +----+-----+------+
        |2012|20000|  5000|
        |2013|30000|  NULL|
        +----+-----+------+
        """
        from krishiv.errors import AnalysisException

        if not isinstance(pivot_col, str):
            raise not_expected_type("pivot_col", "str", pivot_col)
        if values is not None:
            if not isinstance(values, list):
                raise not_expected_type("values", "list", values)
            for v in values:
                if not isinstance(v, bool | int | float | str):
                    raise not_expected_type("values", "list of bool, int, float or str", v)
        if self._pivoted:
            raise AnalysisException(
                "[REPEATED_CLAUSE] The PIVOT clause may be used at most once per SUBQUERY operation. SQLSTATE: 42614",
                errorClass="REPEATED_CLAUSE",
                messageParameters={"clause": "PIVOT", "operation": "SUBQUERY"},
                sqlState="42614",
            )
        column = F.col(pivot_col)
        literals = [F.lit(v) for v in values] if values else self._distinct_values(pivot_col)
        pivoted = GroupedData(self._grouped.pivot(column._c, [v._c for v in literals]), self._source)
        pivoted._pivoted = True
        return pivoted

    def _distinct_values(self, pivot_col: str) -> list[Column]:
        """The values of ``pivot_col``, ascending with NULL first, as Spark's ``collectPivotValues``."""
        from krishiv.errors import AnalysisException
        from krishiv.sql.stat import PIVOT_MAX_VALUES_KEY, _pivot_max_values

        df = self._source
        limit = _pivot_max_values(df.sparkSession)
        column = F.col(pivot_col)
        rows = df.select(column).distinct().limit(limit + 1).sort(column).collect()
        if len(rows) > limit:
            raise AnalysisException(
                f"[_LEGACY_ERROR_TEMP_1324] The pivot column {pivot_col} has more than {limit} distinct values, "
                f"this could indicate an error. If this was intended, set {PIVOT_MAX_VALUES_KEY} to at least "
                "the number of distinct values of the pivot column.",
                errorClass="_LEGACY_ERROR_TEMP_1324",
                messageParameters={"pivotColumn": pivot_col, "maxValues": str(limit), "config": PIVOT_MAX_VALUES_KEY},
            )
        out: list[Column] = []
        seen_zero = seen_nan = False
        for (value,) in rows:
            # -0.0 and 0.0 are one value, and so are the NaNs.
            if isinstance(value, float) and value == 0.0:
                if seen_zero:
                    continue
                seen_zero = True
                value = 0.0
            elif isinstance(value, float) and value != value:
                if seen_nan:
                    continue
                seen_nan = True
            out.append(F.lit(value))
        return out
