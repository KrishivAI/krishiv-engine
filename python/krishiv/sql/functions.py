# SPDX-License-Identifier: Apache-2.0
"""``pyspark.sql.functions``: column builders. A function resolves by name in the engine's
function registry when the query is analyzed, so one that is not registered fails there.

The wrappers of the registered functions are generated (``_generated_functions``, from the registry and
PySpark's signatures) and re-exported here; the ones that are not plain calls are written below.
``__all__`` is the generated names plus ``_HAND_WRITTEN``, so a new registry function needs no edit here.

A name in ``__all__`` is a function with PySpark's signature that builds the registry call; it does not mean the
engine runs every one end to end. The ranking and window-only functions (``row_number``, ``rank``,
``dense_rank``, ``cume_dist``, ``percent_rank``, ``ntile``, ``first_value``, ``last_value``, ``lag``, ``lead``,
``nth_value``) are exported, but
``Column`` has no ``over`` yet, so they cannot be used. ``coalesce`` / ``nvl`` with a string literal and
``sha2`` over a string fail in the engine's analyzer."""

from __future__ import annotations

from typing import Any

from krishiv import _native
from krishiv.errors import not_expected_type
from krishiv.sql import _generated_functions
from krishiv.sql._function_support import ColumnOrName, _call, _named, _num_or_col
from krishiv.sql._generated_functions import *  # noqa: F403  # re-exported through `__all__`
from krishiv.sql.column import Column, _literal, _operand

#: The functions written by hand below; every other name in ``__all__`` is generated.
_HAND_WRITTEN = [
    "asc",
    "call_function",
    "col",
    "column",
    "concat",
    "count",
    "countDistinct",
    "count_distinct",
    "desc",
    "expr",
    "lit",
    "log",
    "when",
]

__all__ = [*_generated_functions.__all__, *_HAND_WRITTEN]

#: The builtin ``str``, kept under its own name: :func:`expr` names its
#: parameter ``str`` (as PySpark does), which shadows the builtin there.
_str = str


def col(col: str) -> Column:
    """The column named ``col`` (``a``, ``t.a``, ``*``)."""
    return Column(_native.col(col))


column = col


def lit(col: Any) -> Column:
    """A literal (``None``, ``bool``, ``int``, ``float``, ``str``); a Column is returned as is. ``lit(None)`` is
    an untyped null (Spark's ``void``)."""
    return col if isinstance(col, Column) else Column(_literal(col))


def expr(str: str) -> Column:
    """Parses the expression string into the column that it represents, as ``pyspark.sql.functions.expr``
    (names resolve against the DataFrame's columns when the query is analyzed)."""
    if not isinstance(str, _str):
        raise not_expected_type("str", "str", str)
    return Column(_native.expr(str))


def call_function(funcName: str, *cols: Any) -> Column:
    """A call of the registered function ``funcName``."""
    return _call(funcName, *[_named(c, "col") for c in cols])


def count(col: ColumnOrName) -> Column:
    """The number of rows where ``col`` is not null (``count("*")``: all rows)."""
    if isinstance(col, str) and col == "*":
        return Column(_native.count_star())
    return _call("count", _named(col, "col"))


def count_distinct(col: ColumnOrName, *cols: ColumnOrName) -> Column:
    return Column(_native.count_distinct([_named(c, "col") for c in (col, *cols)]))


countDistinct = count_distinct


def log(arg1: ColumnOrName | float, arg2: ColumnOrName | None = None) -> Column:
    """The natural logarithm of ``arg1``, or with ``arg2`` the logarithm of ``arg2`` to the base ``arg1``."""
    if arg2 is None:
        return _call("ln", _named(arg1, "arg1"))
    return _call("log", _num_or_col(arg1, "arg1"), _named(arg2, "arg2"))


def concat(*cols: ColumnOrName) -> Column:
    return _call("concat", *[_named(c, "cols") for c in cols])


def when(condition: Column, value: Any) -> Column:
    """Starts a ``CASE WHEN condition THEN value`` chain, as
    ``pyspark.sql.functions.when``. Chain :meth:`Column.when` for more
    branches and :meth:`Column.otherwise` for the ``ELSE`` value; without
    ``otherwise`` unmatched rows are NULL.

    Examples
    --------
    >>> df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    >>> df.select(df.name, when(df.age > 3, 1).otherwise(0)).show()
    +-----+-------------------------------------+
    | name|CASE WHEN (age > 3) THEN 1 ELSE 0 END|
    +-----+-------------------------------------+
    |Alice|                                    0|
    |  Bob|                                    1|
    +-----+-------------------------------------+
    """
    from krishiv.sql.column import _WhenColumn

    if not isinstance(condition, Column):
        raise not_expected_type("condition", "Column", condition)
    return _WhenColumn([(condition._c, _operand(value))])


def asc(col: ColumnOrName) -> Column:
    return Column(_named(col, "col")).asc()


def desc(col: ColumnOrName) -> Column:
    return Column(_named(col, "col")).desc()
