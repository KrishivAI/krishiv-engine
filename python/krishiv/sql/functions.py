# SPDX-License-Identifier: Apache-2.0
"""``pyspark.sql.functions``: column builders. A function resolves by name in the engine's
function registry when the query is analyzed, so one that is not registered fails there.

The wrappers of the registered functions are generated (``_generated_functions``, from the registry and
PySpark's signatures) and re-exported here; the ones that are not plain calls are written below.
``__all__`` is the generated names plus ``_HAND_WRITTEN``, so a new registry function needs no edit here.

A name in ``__all__`` is a function with PySpark's signature that builds the registry call; it does not mean the
engine runs every one end to end. The ranking and window functions (``row_number``, ``rank``,
``dense_rank``, ``cume_dist``, ``percent_rank``, ``ntile``, ``first_value``, ``last_value``, ``lag``, ``lead``,
``nth_value``) run over a window with :meth:`krishiv.sql.Column.over`. ``coalesce`` / ``nvl`` with a string
literal and ``sha2`` over a string fail in the engine's analyzer."""

from __future__ import annotations

from typing import Any

from krishiv import _native
from krishiv.errors import not_expected_type
from krishiv.sql import _generated_functions
from krishiv.sql._function_support import ColumnOrName, _call, _lit, _named, _num_or_col
from krishiv.sql._generated_functions import *  # noqa: F403  # re-exported through `__all__`
from krishiv.sql.column import Column, _literal, _operand
from krishiv.sql.pandas.functions import ArrowUDFType, PandasUDFType, arrow_udf, pandas_udf  # noqa: F401  # re-exported through `__all__`
from krishiv.sql.types import DataType, StringType

#: The functions written by hand below; every other name in ``__all__`` is generated.
_HAND_WRITTEN = [
    "ArrowUDFType",
    "PandasUDFType",
    "arrow_udf",
    "asc",
    "call_function",
    "col",
    "column",
    "concat",
    "convert_timezone",
    "make_dt_interval",
    "make_interval",
    "make_ym_interval",
    "pandas_udf",
    "try_make_interval",
    "udf",
    "count",
    "countDistinct",
    "count_distinct",
    "desc",
    "expr",
    "lit",
    "log",
    "unix_timestamp",
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


def unix_timestamp(timestamp: ColumnOrName | None = None, format: str = "yyyy-MM-dd HH:mm:ss") -> Column:
    """The current Unix time in seconds, or ``timestamp`` parsed with ``format``.

    Without ``timestamp`` the call has no arguments (as in PySpark): the default ``format`` is not a timestamp.
    """
    if timestamp is None:
        return _call("unix_timestamp")
    return _call("unix_timestamp", _named(timestamp, "timestamp"), _lit(format, "format"))


def convert_timezone(sourceTz: Column | None, targetTz: Column, sourceTs: ColumnOrName) -> Column:
    """Converts ``sourceTs`` from the ``sourceTz`` time zone to ``targetTz``.

    When ``sourceTz`` is None the current session time zone is used.
    """
    if sourceTz is None:
        return _call("convert_timezone", _lit(targetTz, "targetTz"), _named(sourceTs, "sourceTs"))
    return _call(
        "convert_timezone",
        _lit(sourceTz, "sourceTz"),
        _lit(targetTz, "targetTz"),
        _named(sourceTs, "sourceTs"),
    )


def _field_or_zero(col: ColumnOrName | None, name: str) -> _native.Column:
    return _literal(0) if col is None else _named(col, name)


def make_ym_interval(years: ColumnOrName | None = None, months: ColumnOrName | None = None) -> Column:
    """A year-month interval; an omitted field is zero."""
    return _call("make_ym_interval", _field_or_zero(years, "years"), _field_or_zero(months, "months"))


def make_dt_interval(
    days: ColumnOrName | None = None,
    hours: ColumnOrName | None = None,
    mins: ColumnOrName | None = None,
    secs: ColumnOrName | None = None,
) -> Column:
    """A day-time interval; an omitted field is zero."""
    return _call(
        "make_dt_interval",
        _field_or_zero(days, "days"),
        _field_or_zero(hours, "hours"),
        _field_or_zero(mins, "mins"),
        _field_or_zero(secs, "secs"),
    )


def make_interval(
    years: ColumnOrName | None = None,
    months: ColumnOrName | None = None,
    weeks: ColumnOrName | None = None,
    days: ColumnOrName | None = None,
    hours: ColumnOrName | None = None,
    mins: ColumnOrName | None = None,
    secs: ColumnOrName | None = None,
) -> Column:
    """A calendar interval; an omitted field is zero."""
    return _call(
        "make_interval",
        _field_or_zero(years, "years"),
        _field_or_zero(months, "months"),
        _field_or_zero(weeks, "weeks"),
        _field_or_zero(days, "days"),
        _field_or_zero(hours, "hours"),
        _field_or_zero(mins, "mins"),
        _field_or_zero(secs, "secs"),
    )


def try_make_interval(
    years: ColumnOrName | None = None,
    months: ColumnOrName | None = None,
    weeks: ColumnOrName | None = None,
    days: ColumnOrName | None = None,
    hours: ColumnOrName | None = None,
    mins: ColumnOrName | None = None,
    secs: ColumnOrName | None = None,
) -> Column:
    """``make_interval``, NULL on overflow; an omitted field is zero."""
    return _call(
        "try_make_interval",
        _field_or_zero(years, "years"),
        _field_or_zero(months, "months"),
        _field_or_zero(weeks, "weeks"),
        _field_or_zero(days, "days"),
        _field_or_zero(hours, "hours"),
        _field_or_zero(mins, "mins"),
        _field_or_zero(secs, "secs"),
    )


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


def udf(
    f: Any = None,
    returnType: DataType | str = StringType(),  # noqa: B008  # Spark's default, an immutable type
    *,
    useArrow: bool | None = None,
) -> Any:
    """Creates a user defined function (``pyspark.sql.functions.udf``, PySpark 4.2.0
    ``sql/functions/builtin.py:30428-30635``): ``udf(f, returnType)``, ``@udf``, ``@udf(returnType)`` or
    ``@udf(returnType=...)``. ``returnType`` defaults to ``StringType``; ``useArrow=None`` follows
    ``spark.sql.execution.pythonUDF.arrow.enabled`` and the function's type hints (``pd.Series -> pd.Series`` is
    a pandas UDF, ``pa.Array -> pa.Array`` an Arrow UDF). Calling the result gives a Column named
    ``<function name>(<args>)`` (``<lambda>`` for a lambda); it is not callable from SQL until
    ``spark.udf.register``.

    Only the Arrow paths are implemented: a UDF Spark would run pickled (``useArrow=False``) runs Arrow-batched
    with a warning (a documented divergence)."""
    import functools

    from krishiv.sql.udf import _create_py_udf

    # decorator @udf, @udf(), @udf(dataType())
    if f is None or isinstance(f, str | DataType):
        # If DataType has been passed as a positional argument
        # for decorator use it as a returnType
        return_type = f or returnType
        return functools.partial(
            _create_py_udf,
            returnType=return_type,
            useArrow=useArrow,
        )
    else:
        return _create_py_udf(f=f, returnType=returnType, useArrow=useArrow)
