# SPDX-License-Identifier: Apache-2.0
"""``pyspark.sql.Window`` and ``pyspark.sql.WindowSpec``: windows for ``Column.over``.

Every method builds the same expression the SQL path builds, resolved by
the analyzer when the query runs. Nothing is evaluated in Python: frames
the engine cannot run yet fail with the engine's own error, as the SQL
does.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from krishiv import _native
from krishiv.errors import not_expected_type
from krishiv.sql._function_support import ColumnOrName
from krishiv.sql.column import Column

#: The JVM ``long`` bounds, as ``pyspark.util.JVM_LONG_MIN/MAX``.
_JVM_LONG_MIN = -(2**63)
_JVM_LONG_MAX = 2**63 - 1


def _cols(cols: tuple[ColumnOrName | Sequence[ColumnOrName], ...]) -> list[_native.Column]:
    """The native columns of ``partitionBy``/``orderBy`` arguments (a single list is flattened, as in
    PySpark)."""
    if len(cols) == 1 and isinstance(cols[0], list):
        cols = tuple(cols[0])
    out = []
    for c in cols:
        if isinstance(c, Column):
            out.append(c._c)
        elif isinstance(c, str):
            out.append(_native.col(c))
        else:
            raise not_expected_type("cols", "Column, str or list of Column or str", c)
    return out


def _bounds(start: int, end: int) -> tuple[int, int]:
    """``rowsBetween``/``rangeBetween`` bounds after PySpark's clamping: at or past the thresholds they
    are the unbounded sentinels."""
    if not isinstance(start, int):
        raise not_expected_type("start", "int", start)
    if not isinstance(end, int):
        raise not_expected_type("end", "int", end)
    if start <= Window._PRECEDING_THRESHOLD:
        start = Window.unboundedPreceding
    if end >= Window._FOLLOWING_THRESHOLD:
        end = Window.unboundedFollowing
    return start, end


class Window:
    """Utility functions for defining windows in DataFrames, as ``pyspark.sql.Window``.

    When ordering is not defined, an unbounded window frame (rowFrame,
    unboundedPreceding, unboundedFollowing) is used by default. When ordering is defined,
    a growing window frame (rangeFrame, unboundedPreceding, currentRow) is used by default.
    """

    _PRECEDING_THRESHOLD = max(-sys.maxsize, _JVM_LONG_MIN)

    _FOLLOWING_THRESHOLD = min(sys.maxsize, _JVM_LONG_MAX)

    unboundedPreceding: int = _JVM_LONG_MIN

    unboundedFollowing: int = _JVM_LONG_MAX

    currentRow: int = 0

    @staticmethod
    def partitionBy(*cols: ColumnOrName | Sequence[ColumnOrName]) -> WindowSpec:
        """Creates a :class:`WindowSpec` with the partitioning defined.

        Parameters
        ----------
        cols : str, :class:`Column` or list
            names of columns or expressions

        Returns
        -------
        :class: `WindowSpec`
            A :class:`WindowSpec` with the partitioning defined.

        Examples
        --------
        >>> from krishiv.sql import Window, functions as sf
        >>> df = spark.createDataFrame(
        ...      [(1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")], ["id", "category"])
        >>> window = Window.partitionBy("category").orderBy("id")
        >>> df.withColumn("row_number", sf.row_number().over(window)).show()
        +---+--------+----------+
        | id|category|row_number|
        +---+--------+----------+
        |  1|       a|         1|
        |  1|       a|         2|
        |  2|       a|         3|
        |  1|       b|         1|
        |  2|       b|         2|
        |  3|       b|         3|
        +---+--------+----------+
        """
        return WindowSpec(_native.WindowSpec()).partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: ColumnOrName | Sequence[ColumnOrName]) -> WindowSpec:
        """Creates a :class:`WindowSpec` with the ordering defined.

        Parameters
        ----------
        cols : str, :class:`Column` or list
            names of columns or expressions

        Returns
        -------
        :class: `WindowSpec`
            A :class:`WindowSpec` with the ordering defined.

        Examples
        --------
        >>> from krishiv.sql import Window, functions as sf
        >>> df = spark.createDataFrame(
        ...      [(1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")], ["id", "category"])
        >>> window = Window.partitionBy("id").orderBy("category")
        >>> df.withColumn("row_number", sf.row_number().over(window)).show()
        +---+--------+----------+
        | id|category|row_number|
        +---+--------+----------+
        |  1|       a|         1|
        |  1|       a|         2|
        |  1|       b|         3|
        |  2|       a|         1|
        |  2|       b|         2|
        |  3|       b|         1|
        +---+--------+----------+
        """
        return WindowSpec(_native.WindowSpec()).orderBy(*cols)

    @staticmethod
    def rowsBetween(start: int, end: int) -> WindowSpec:
        """Creates a :class:`WindowSpec` with the frame boundaries defined,
        from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative positions from the current row.
        For example, "0" means "current row", while "-1" means the row before
        the current row, and "5" means the fifth row after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        Parameters
        ----------
        start : int
            boundary start, inclusive.
            The frame is unbounded if this is ``Window.unboundedPreceding``, or
            any value less than or equal to -9223372036854775808.
        end : int
            boundary end, inclusive.
            The frame is unbounded if this is ``Window.unboundedFollowing``, or
            any value greater than or equal to 9223372036854775807.

        Returns
        -------
        :class: `WindowSpec`
            A :class:`WindowSpec` with the frame boundaries defined,
            from `start` (inclusive) to `end` (inclusive).

        Examples
        --------
        >>> from krishiv.sql import Window, functions as sf
        >>> df = spark.createDataFrame(
        ...      [(1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")], ["id", "category"])
        >>> window = Window.partitionBy("category").orderBy("id").rowsBetween(Window.currentRow, 1)
        >>> df.withColumn("sum", sf.sum("id").over(window)).sort("id", "category", "sum").show()
        +---+--------+---+
        | id|category|sum|
        +---+--------+---+
        |  1|       a|  2|
        |  1|       a|  3|
        |  1|       b|  3|
        |  2|       a|  2|
        |  2|       b|  5|
        |  3|       b|  3|
        +---+--------+---+
        """
        return WindowSpec(_native.WindowSpec()).rowsBetween(start, end)

    @staticmethod
    def rangeBetween(start: int, end: int) -> WindowSpec:
        """Creates a :class:`WindowSpec` with the frame boundaries defined,
        from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative from the current row. For example,
        "0" means "current row", while "-1" means one off before the current row,
        and "5" means the five off after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        A range-based boundary is based on the actual value of the ORDER BY
        expression(s).

        Parameters
        ----------
        start : int
            boundary start, inclusive.
            The frame is unbounded if this is ``Window.unboundedPreceding``, or
            any value less than or equal to max(-sys.maxsize, -9223372036854775808).
        end : int
            boundary end, inclusive.
            The frame is unbounded if this is ``Window.unboundedFollowing``, or
            any value greater than or equal to min(sys.maxsize, 9223372036854775807).

        Returns
        -------
        :class: `WindowSpec`
            A :class:`WindowSpec` with the frame boundaries defined,
            from `start` (inclusive) to `end` (inclusive).

        .. note:: The engine has no window frames with value bounds yet:
            such a frame fails with the engine's own
            ``KRISHIV_UNSUPPORTED_FEATURE`` error, as the SQL form does.

        Examples
        --------
        >>> from krishiv.sql import Window, functions as sf
        >>> df = spark.createDataFrame(
        ...      [(1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")], ["id", "category"])
        >>> window = Window.partitionBy("category").orderBy("id").rangeBetween(Window.currentRow, 1)
        >>> df.withColumn("sum", sf.sum("id").over(window)).sort("id", "category").show()
        +---+--------+---+
        | id|category|sum|
        +---+--------+---+
        |  1|       a|  4|
        |  1|       a|  4|
        |  1|       b|  3|
        |  2|       a|  2|
        |  2|       b|  5|
        |  3|       b|  3|
        +---+--------+---+
        """
        return WindowSpec(_native.WindowSpec()).rangeBetween(start, end)


class WindowSpec:
    """A window specification that defines the partitioning, ordering,
    and frame boundaries, as ``pyspark.sql.WindowSpec``.

    Use the static methods in :class:`Window` to create a :class:`WindowSpec`.
    Each method returns a new specification; a repeated ``partitionBy`` or
    ``orderBy`` replaces what the earlier call set, as in PySpark.
    """

    __slots__ = ("_s",)

    def __init__(self, native: _native.WindowSpec) -> None:
        self._s = native

    def partitionBy(self, *cols: ColumnOrName | Sequence[ColumnOrName]) -> WindowSpec:
        """Defines the partitioning columns in a :class:`WindowSpec`, replacing any partitioning.

        Parameters
        ----------
        cols : str, :class:`Column` or list
            names of columns or expressions
        """
        return WindowSpec(self._s.partition_by(_cols(cols)))

    def orderBy(self, *cols: ColumnOrName | Sequence[ColumnOrName]) -> WindowSpec:
        """Defines the ordering columns in a :class:`WindowSpec`, replacing any ordering.

        Parameters
        ----------
        cols : str, :class:`Column` or list
            names of columns or expressions
        """
        return WindowSpec(self._s.order_by(_cols(cols)))

    def rowsBetween(self, start: int, end: int) -> WindowSpec:
        """Defines the frame boundaries, from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative positions from the current row.
        For example, "0" means "current row", while "-1" means the row before
        the current row, and "5" means the fifth row after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        Parameters
        ----------
        start : int
            boundary start, inclusive.
            The frame is unbounded if this is ``Window.unboundedPreceding``, or
            any value less than or equal to max(-sys.maxsize, -9223372036854775808).
        end : int
            boundary end, inclusive.
            The frame is unbounded if this is ``Window.unboundedFollowing``, or
            any value greater than or equal to min(sys.maxsize, 9223372036854775807).
        """
        start, end = _bounds(start, end)
        return WindowSpec(self._s.rows_between(start, end))

    def rangeBetween(self, start: int, end: int) -> WindowSpec:
        """Defines the frame boundaries, from `start` (inclusive) to `end` (inclusive).

        Both `start` and `end` are relative from the current row. For example,
        "0" means "current row", while "-1" means one off before the current row,
        and "5" means the five off after the current row.

        We recommend users use ``Window.unboundedPreceding``, ``Window.unboundedFollowing``,
        and ``Window.currentRow`` to specify special boundary values, rather than using integral
        values directly.

        Parameters
        ----------
        start : int
            boundary start, inclusive.
            The frame is unbounded if this is ``Window.unboundedPreceding``, or
            any value less than or equal to max(-sys.maxsize, -9223372036854775808).
        end : int
            boundary end, inclusive.
            The frame is unbounded if this is ``Window.unboundedFollowing``, or
            any value greater than or equal to min(sys.maxsize, 9223372036854775807).

        .. note:: The engine has no window frames with value bounds yet:
            such a frame fails with the engine's own
            ``KRISHIV_UNSUPPORTED_FEATURE`` error, as the SQL form does.
        """
        start, end = _bounds(start, end)
        return WindowSpec(self._s.range_between(start, end))
