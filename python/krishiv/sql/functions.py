# SPDX-License-Identifier: Apache-2.0
"""``pyspark.sql.functions``: column builders. A function resolves by name in the engine's
function registry when the query is analyzed, so one that is not registered fails there."""

from __future__ import annotations

from typing import Any

from krishiv import _native
from krishiv.errors import not_implemented
from krishiv.sql.column import Column, _literal, _named

ColumnOrName = Column | str


def col(col: str) -> Column:
    """The column named ``col`` (``a``, ``t.a``, ``*``)."""
    return Column(_native.col(col))


column = col


def lit(col: Any) -> Column:
    """A literal (``bool``, ``int``, ``float``, ``str``); a Column is returned as is."""
    return col if isinstance(col, Column) else Column(_literal(col))


def expr(str: str) -> Column:
    raise not_implemented("expr (SQL expression strings)")


def _call(name: str, *cols: Any) -> Column:
    return Column(_native.call_function(name, [_named(c, "col") for c in cols]))


def call_function(funcName: str, *cols: Any) -> Column:
    """A call of the registered function ``funcName``."""
    return _call(funcName, *cols)


def _unary(name: str, doc: str) -> Any:
    def f(col: ColumnOrName) -> Column:
        return _call(name, col)

    f.__name__ = name
    f.__qualname__ = name
    f.__doc__ = doc
    return f


# Aggregates.
sum = _unary("sum", "The sum of the group.")
avg = _unary("avg", "The average of the group.")
min = _unary("min", "The minimum of the group.")
max = _unary("max", "The maximum of the group.")


def mean(col: ColumnOrName) -> Column:
    """Alias of :func:`avg`."""
    return avg(col)


def count(col: ColumnOrName) -> Column:
    """The number of rows where ``col`` is not null (``count("*")``: all rows)."""
    if isinstance(col, str) and col == "*":
        return Column(_native.count_star())
    return _call("count", col)


def count_distinct(col: ColumnOrName, *cols: ColumnOrName) -> Column:
    return Column(_native.count_distinct([_named(c, "col") for c in (col, *cols)]))


countDistinct = count_distinct


# Scalar functions (a subset; the registry decides which exist).
abs = _unary("abs", "The absolute value.")
sqrt = _unary("sqrt", "The square root.")
floor = _unary("floor", "The largest integer not above the value.")
ceil = _unary("ceil", "The smallest integer not below the value.")
upper = _unary("upper", "The string in upper case.")
lower = _unary("lower", "The string in lower case.")
length = _unary("length", "The length of the string or binary.")
trim = _unary("trim", "The string without surrounding spaces.")
ltrim = _unary("ltrim", "The string without leading spaces.")
rtrim = _unary("rtrim", "The string without trailing spaces.")


def coalesce(*cols: ColumnOrName) -> Column:
    """The first column that is not null."""
    return _call("coalesce", *cols)


def concat(*cols: ColumnOrName) -> Column:
    return _call("concat", *cols)


def greatest(*cols: ColumnOrName) -> Column:
    return _call("greatest", *cols)


def least(*cols: ColumnOrName) -> Column:
    return _call("least", *cols)


def asc(col: ColumnOrName) -> Column:
    return Column(_named(col, "col")).asc()


def desc(col: ColumnOrName) -> Column:
    return Column(_named(col, "col")).desc()
