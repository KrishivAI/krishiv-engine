# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403, F405  # the generated names arrive by `import *` and are re-exported through `__all__`
"""``pyspark.sql.functions``: column builders. A function resolves by name in the engine's
function registry when the query is analyzed, so one that is not registered fails there.

The wrappers of the registered functions are generated (``_generated_functions``, from the registry and
PySpark's signatures) and re-exported here; the ones that are not plain calls are written below."""

from __future__ import annotations

from typing import Any

from krishiv import _native
from krishiv.errors import not_expected_type
from krishiv.sql._function_support import ColumnOrName, _call, _named, _num_or_col
from krishiv.sql._generated_functions import *
from krishiv.sql.column import Column, _literal, _operand

__all__ = [
    "abs",
    "acos",
    "acosh",
    "asc",
    "asin",
    "asinh",
    "assert_true",
    "atan",
    "atan2",
    "atanh",
    "avg",
    "bin",
    "bit_count",
    "bit_get",
    "bitwiseNOT",
    "bitwise_not",
    "bround",
    "btrim",
    "call_function",
    "cbrt",
    "ceil",
    "ceiling",
    "char_length",
    "character_length",
    "coalesce",
    "col",
    "column",
    "concat",
    "contains",
    "conv",
    "cos",
    "cosh",
    "cot",
    "count",
    "countDistinct",
    "count_distinct",
    "crc32",
    "csc",
    "cume_dist",
    "curdate",
    "current_date",
    "current_timestamp",
    "current_user",
    "date_add",
    "date_diff",
    "date_part",
    "date_sub",
    "dateadd",
    "datediff",
    "day",
    "dayofmonth",
    "degrees",
    "dense_rank",
    "desc",
    "e",
    "endswith",
    "equal_null",
    "exp",
    "expm1",
    "expr",
    "extract",
    "first",
    "first_value",
    "floor",
    "format_number",
    "getbit",
    "greatest",
    "hash",
    "hex",
    "hypot",
    "ifnull",
    "isnan",
    "isnotnull",
    "isnull",
    "last",
    "last_value",
    "lcase",
    "least",
    "length",
    "lit",
    "ln",
    "localtimestamp",
    "log",
    "log1p",
    "log2",
    "log10",
    "lower",
    "lpad",
    "ltrim",
    "max",
    "md5",
    "mean",
    "min",
    "month",
    "nanvl",
    "negative",
    "now",
    "ntile",
    "nullif",
    "nullifzero",
    "nvl",
    "nvl2",
    "percent_rank",
    "pi",
    "pmod",
    "positive",
    "pow",
    "radians",
    "raise_error",
    "rank",
    "regexp",
    "regexp_count",
    "regexp_extract",
    "regexp_instr",
    "regexp_like",
    "regexp_replace",
    "regexp_substr",
    "replace",
    "rint",
    "round",
    "row_number",
    "rpad",
    "rtrim",
    "sec",
    "session_user",
    "sha",
    "sha1",
    "sha2",
    "shiftLeft",
    "shiftRight",
    "shiftRightUnsigned",
    "shiftleft",
    "shiftright",
    "shiftrightunsigned",
    "sign",
    "signum",
    "sin",
    "sinh",
    "sqrt",
    "startswith",
    "substr",
    "substring",
    "sum",
    "tan",
    "tanh",
    "toDegrees",
    "toRadians",
    "to_char",
    "to_number",
    "to_varchar",
    "trim",
    "try_add",
    "try_divide",
    "try_mod",
    "try_multiply",
    "try_subtract",
    "try_to_number",
    "ucase",
    "unhex",
    "upper",
    "user",
    "when",
    "xxhash64",
    "year",
    "zeroifnull",
]

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
