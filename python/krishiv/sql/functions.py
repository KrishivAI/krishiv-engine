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

import inspect
import itertools
import secrets
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from krishiv import _native
from krishiv.errors import PySparkValueError, not_expected_type, not_implemented
from krishiv.sql import _generated_functions
from krishiv.sql._function_support import ColumnOrName, _call, _lit, _named, _num_or_col
from krishiv.sql._generated_functions import *  # noqa: F403  # re-exported through `__all__`
from krishiv.sql.column import Column, _literal, _operand
from krishiv.sql.pandas.functions import ArrowUDFType, PandasUDFType, arrow_udf, pandas_udf  # noqa: F401  # re-exported through `__all__`
from krishiv.sql.types import DataType, StringType

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame

#: The functions written by hand below; every other name in ``__all__`` is generated.
_HAND_WRITTEN = [
    "ArrowUDFType",
    "PandasUDFType",
    "aggregate",
    "array_sort",
    "arrow_udf",
    "arrow_udtf",
    "asc",
    "asc_nulls_first",
    "asc_nulls_last",
    "broadcast",
    "bucket",
    "call_function",
    "call_udf",
    "col",
    "column",
    "concat",
    "convert_timezone",
    "count",
    "countDistinct",
    "count_distinct",
    "current_time",
    "days",
    "desc",
    "desc_nulls_first",
    "desc_nulls_last",
    "exists",
    "explode",
    "explode_outer",
    "expr",
    "filter",
    "forall",
    "hours",
    "inline",
    "inline_outer",
    "listagg_distinct",
    "lit",
    "log",
    "make_dt_interval",
    "make_interval",
    "make_time",
    "make_ym_interval",
    "map_filter",
    "map_zip_with",
    "months",
    "pandas_udf",
    "posexplode",
    "posexplode_outer",
    "product",
    "reduce",
    "st_asbinary",
    "st_geogfromwkb",
    "st_geomfromwkb",
    "st_setsrid",
    "st_srid",
    "stack",
    "string_agg_distinct",
    "sumDistinct",
    "sum_distinct",
    "time_diff",
    "time_from_micros",
    "time_from_millis",
    "time_from_seconds",
    "time_to_micros",
    "time_to_millis",
    "time_to_seconds",
    "time_trunc",
    "to_time",
    "transform",
    "transform_keys",
    "transform_values",
    "try_make_interval",
    "try_to_time",
    "udf",
    "udtf",
    "unix_timestamp",
    "unwrap_udt",
    "when",
    "window",
    "window_time",
    "years",
    "zip_with",
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


_DISTINCT = "DISTINCT aggregates have no Column builder yet; use `F.expr`, e.g. `F.expr('sum(DISTINCT c)')`"

# Generators. Each builds the call that ``explode(col)`` is in SQL; the DataFrame API turns it into a ``Generate``
# node (a generator in ``select`` / ``withColumn``).


def explode(col: ColumnOrName) -> Column:
    """One row per element of the array (or entry of the map) ``col``."""
    return _call("explode", _named(col, "col"))


def explode_outer(col: ColumnOrName) -> Column:
    """As :func:`explode`, and a null or empty ``col`` gives one row of nulls."""
    return _call("explode_outer", _named(col, "col"))


def posexplode(col: ColumnOrName) -> Column:
    """As :func:`explode`, with the position of the element (``pos``, ``col``; ``pos``, ``key``, ``value``)."""
    return _call("posexplode", _named(col, "col"))


def posexplode_outer(col: ColumnOrName) -> Column:
    """As :func:`posexplode`, and a null or empty ``col`` gives one row of nulls."""
    return _call("posexplode_outer", _named(col, "col"))


def inline(col: ColumnOrName) -> Column:
    """One row per struct of the array of structs ``col``, one column per field."""
    return _call("inline", _named(col, "col"))


def inline_outer(col: ColumnOrName) -> Column:
    """As :func:`inline`, and a null or empty ``col`` gives one row of nulls."""
    return _call("inline_outer", _named(col, "col"))


def stack(*cols: ColumnOrName) -> Column:
    """Separates ``cols[1:]`` into ``cols[0]`` rows (``cols[0]`` is the row count, a literal column)."""
    return _call("stack", *[_named(c, "cols") for c in cols])


# Higher-order functions: the Python function is called once with placeholder columns, and the Column it returns
# becomes the SQL lambda (``x -> x + 1``), as in PySpark.

_lambda_ids = itertools.count()
#: Lambda variables are columns by name to the engine, so theirs carry a prefix no user column has (and a
#: token that differs per process): ``lambda x: x + F.col("x_0")`` must read the column ``x_0``.
_LAMBDA_TOKEN = secrets.token_hex(4)


def _lambda(f: Callable[..., Column]) -> _native.Column:
    """The lambda that the Python function ``f`` (one to three parameters, as in PySpark) stands for. A function
    with the wrong number of parameters for the higher-order function fails when the query is analyzed."""
    func_name = getattr(f, "__name__", None) or str(f)
    if not callable(f):
        raise not_expected_type("f", "function", f)
    params = list(inspect.signature(f).parameters.values())
    if not all(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params):
        raise PySparkValueError(
            f"[UNSUPPORTED_PARAM_TYPE_FOR_HIGHER_ORDER_FUNCTION] Function `{func_name}` should use only POSITIONAL "
            "or POSITIONAL OR KEYWORD arguments.",
            errorClass="UNSUPPORTED_PARAM_TYPE_FOR_HIGHER_ORDER_FUNCTION",
            messageParameters={"func_name": func_name},
        )
    if not 1 <= len(params) <= 3:
        raise PySparkValueError(
            f"[WRONG_NUM_ARGS_FOR_HIGHER_ORDER_FUNCTION] Function `{func_name}` should take between 1 and 3 "
            f"arguments, but the provided function takes {len(params)}.",
            errorClass="WRONG_NUM_ARGS_FOR_HIGHER_ORDER_FUNCTION",
            messageParameters={"func_name": func_name, "num_args": str(len(params))},
        )
    # Fresh names, so a lambda inside a lambda never captures its parent's variable.
    names = [f"__krishiv_lambda_{_LAMBDA_TOKEN}_{next(_lambda_ids)}_{base}" for base in ("x", "y", "z")[: len(params)]]
    body = f(*[Column(_native.col(n)) for n in names])
    if not isinstance(body, Column):
        raise PySparkValueError(
            f"[HIGHER_ORDER_FUNCTION_SHOULD_RETURN_COLUMN] Function `{func_name}` should return Column, "
            f"got {type(body)}.",
            errorClass="HIGHER_ORDER_FUNCTION_SHOULD_RETURN_COLUMN",
            messageParameters={"func_name": func_name, "return_type": str(type(body))},
        )
    return _native.lambda_function(names, body._c)


def transform(col: ColumnOrName, f: Callable[..., Column]) -> Column:
    """The array ``col`` with ``f(element)`` or ``f(element, index)`` of each element."""
    return _call("transform", _named(col, "col"), _lambda(f))


def exists(col: ColumnOrName, f: Callable[[Column], Column]) -> Column:
    """Whether ``f(element)`` holds for at least one element of the array ``col``."""
    return _call("exists", _named(col, "col"), _lambda(f))


def forall(col: ColumnOrName, f: Callable[[Column], Column]) -> Column:
    """Whether ``f(element)`` holds for every element of the array ``col``."""
    return _call("forall", _named(col, "col"), _lambda(f))


def filter(col: ColumnOrName, f: Callable[..., Column]) -> Column:  # PySpark's name
    """The elements of the array ``col`` for which ``f(element)`` or ``f(element, index)`` holds."""
    return _call("filter", _named(col, "col"), _lambda(f))


def aggregate(
    col: ColumnOrName,
    initialValue: ColumnOrName,
    merge: Callable[[Column, Column], Column],
    finish: Callable[[Column], Column] | None = None,
) -> Column:
    """Folds the array ``col`` from ``initialValue`` with ``merge(acc, element)``, then ``finish(acc)`` if given."""
    args = [_named(col, "col"), _named(initialValue, "initialValue"), _lambda(merge)]
    if finish is not None:
        args.append(_lambda(finish))
    return _call("aggregate", *args)


def reduce(
    col: ColumnOrName,
    initialValue: ColumnOrName,
    merge: Callable[[Column, Column], Column],
    finish: Callable[[Column], Column] | None = None,
) -> Column:
    """An alias of :func:`aggregate`."""
    return aggregate(col, initialValue, merge, finish)


def zip_with(left: ColumnOrName, right: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column:
    """The arrays ``left`` and ``right`` merged element by element with ``f(l, r)``."""
    return _call("zip_with", _named(left, "left"), _named(right, "right"), _lambda(f))


def map_filter(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column:
    """The entries of the map ``col`` for which ``f(key, value)`` holds."""
    return _call("map_filter", _named(col, "col"), _lambda(f))


def map_zip_with(col1: ColumnOrName, col2: ColumnOrName, f: Callable[[Column, Column, Column], Column]) -> Column:
    """The maps ``col1`` and ``col2`` merged by key with ``f(key, value1, value2)``."""
    return _call("map_zip_with", _named(col1, "col1"), _named(col2, "col2"), _lambda(f))


def transform_keys(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column:
    """The map ``col`` with the keys ``f(key, value)``."""
    return _call("transform_keys", _named(col, "col"), _lambda(f))


def transform_values(col: ColumnOrName, f: Callable[[Column, Column], Column]) -> Column:
    """The map ``col`` with the values ``f(key, value)``."""
    return _call("transform_values", _named(col, "col"), _lambda(f))


def array_sort(col: ColumnOrName, comparator: Callable[[Column, Column], Column] | None = None) -> Column:
    """The array ``col`` sorted ascending (nulls last), or by ``comparator(left, right)``, which returns a negative
    integer, 0 or a positive integer."""
    if comparator is None:
        return _call("array_sort", _named(col, "col"))
    return _call("array_sort", _named(col, "col"), _lambda(comparator))


def asc_nulls_first(col: ColumnOrName) -> Column:
    """A sort expression: ``col`` ascending, nulls first."""
    return Column(_named(col, "col")).asc_nulls_first()


def asc_nulls_last(col: ColumnOrName) -> Column:
    """A sort expression: ``col`` ascending, nulls last."""
    return Column(_named(col, "col")).asc_nulls_last()


def desc_nulls_first(col: ColumnOrName) -> Column:
    """A sort expression: ``col`` descending, nulls first."""
    return Column(_named(col, "col")).desc_nulls_first()


def desc_nulls_last(col: ColumnOrName) -> Column:
    """A sort expression: ``col`` descending, nulls last."""
    return Column(_named(col, "col")).desc_nulls_last()


def broadcast(df: DataFrame) -> DataFrame:
    """Marks ``df`` as small enough for a broadcast join (``df.hint("broadcast")``)."""
    return df.hint("broadcast")


def call_udf(udfName: str, *cols: ColumnOrName) -> Column:
    """A call of the function ``udfName`` (a registered UDF or any registered function) on ``cols``."""
    return _call(udfName, *[_named(c, "cols") for c in cols])


# The functions below exist under PySpark's names and fail with ``PySparkNotImplementedError`` (a
# ``NotImplementedError``) where the engine, or the Python mapping of a type, is missing.


def product(col: ColumnOrName) -> Column:
    """Not available: the engine has no `product` aggregate."""
    raise not_implemented("F.product (the engine has no `product` aggregate)")


def sum_distinct(col: ColumnOrName) -> Column:
    """Not available: DISTINCT aggregates have no Column builder yet."""
    raise not_implemented(f"F.sum_distinct ({_DISTINCT})")


def sumDistinct(col: ColumnOrName) -> Column:
    """Not available: DISTINCT aggregates have no Column builder yet."""
    raise not_implemented(f"F.sumDistinct ({_DISTINCT})")


def listagg_distinct(col: ColumnOrName, delimiter: Any = None) -> Column:
    """Not available: DISTINCT aggregates have no Column builder yet."""
    raise not_implemented(f"F.listagg_distinct ({_DISTINCT})")


def string_agg_distinct(col: ColumnOrName, delimiter: Any = None) -> Column:
    """Not available: DISTINCT aggregates have no Column builder yet."""
    raise not_implemented(f"F.string_agg_distinct ({_DISTINCT})")


def bucket(numBuckets: Any, col: ColumnOrName) -> Column:
    """Not available: partitioning transforms are not implemented."""
    raise not_implemented("F.bucket (partitioning transforms are not implemented)")


def years(col: ColumnOrName) -> Column:
    """Not available: partitioning transforms are not implemented."""
    raise not_implemented("F.years (partitioning transforms are not implemented)")


def months(col: ColumnOrName) -> Column:
    """Not available: partitioning transforms are not implemented."""
    raise not_implemented("F.months (partitioning transforms are not implemented)")


def days(col: ColumnOrName) -> Column:
    """Not available: partitioning transforms are not implemented."""
    raise not_implemented("F.days (partitioning transforms are not implemented)")


def hours(col: ColumnOrName) -> Column:
    """Not available: partitioning transforms are not implemented."""
    raise not_implemented("F.hours (partitioning transforms are not implemented)")


def udtf(cls: Any = None, *, returnType: Any = None, useArrow: bool | None = None) -> Any:
    """Not available: table UDFs (UDTF) are not implemented."""
    raise not_implemented("F.udtf (table UDFs (UDTF) are not implemented)")


def arrow_udtf(cls: Any = None, *, returnType: Any = None) -> Any:
    """Not available: table UDFs (UDTF) are not implemented."""
    raise not_implemented("F.arrow_udtf (table UDFs (UDTF) are not implemented)")


def unwrap_udt(col: ColumnOrName) -> Column:
    """Not available: user-defined types are not implemented."""
    raise not_implemented("F.unwrap_udt (user-defined types are not implemented)")


def window(
    timeColumn: ColumnOrName, windowDuration: str, slideDuration: str | None = None, startTime: str | None = None
) -> Column:
    """Not available: time windows through the Column API are not implemented."""
    raise not_implemented("F.window (time windows through the Column API are not implemented)")


def window_time(windowColumn: ColumnOrName) -> Column:
    """Not available: time windows through the Column API are not implemented."""
    raise not_implemented("F.window_time (time windows through the Column API are not implemented)")


def current_time(precision: int | None = None) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.current_time (the TIME type has no Python mapping yet (P2-T181))")


def make_time(hour: ColumnOrName, minute: ColumnOrName, second: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.make_time (the TIME type has no Python mapping yet (P2-T181))")


def time_diff(unit: ColumnOrName, start: ColumnOrName, end: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.time_diff (the TIME type has no Python mapping yet (P2-T181))")


def time_from_micros(col: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.time_from_micros (the TIME type has no Python mapping yet (P2-T181))")


def time_from_millis(col: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.time_from_millis (the TIME type has no Python mapping yet (P2-T181))")


def time_from_seconds(col: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.time_from_seconds (the TIME type has no Python mapping yet (P2-T181))")


def time_to_micros(col: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.time_to_micros (the TIME type has no Python mapping yet (P2-T181))")


def time_to_millis(col: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.time_to_millis (the TIME type has no Python mapping yet (P2-T181))")


def time_to_seconds(col: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.time_to_seconds (the TIME type has no Python mapping yet (P2-T181))")


def time_trunc(unit: ColumnOrName, time: ColumnOrName) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.time_trunc (the TIME type has no Python mapping yet (P2-T181))")


def to_time(str: ColumnOrName, format: ColumnOrName | None = None) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.to_time (the TIME type has no Python mapping yet (P2-T181))")


def try_to_time(str: ColumnOrName, format: ColumnOrName | None = None) -> Column:
    """Not available: the TIME type has no Python mapping yet."""
    raise not_implemented("F.try_to_time (the TIME type has no Python mapping yet (P2-T181))")


def st_asbinary(geo: ColumnOrName, endianness: ColumnOrName | None = None) -> Column:
    """Not available: GEOMETRY and GEOGRAPHY have no Python mapping yet."""
    raise not_implemented("F.st_asbinary (GEOMETRY and GEOGRAPHY have no Python mapping yet (P2-T181))")


def st_geogfromwkb(wkb: ColumnOrName) -> Column:
    """Not available: GEOMETRY and GEOGRAPHY have no Python mapping yet."""
    raise not_implemented("F.st_geogfromwkb (GEOMETRY and GEOGRAPHY have no Python mapping yet (P2-T181))")


def st_geomfromwkb(wkb: ColumnOrName, srid: Any = None) -> Column:
    """Not available: GEOMETRY and GEOGRAPHY have no Python mapping yet."""
    raise not_implemented("F.st_geomfromwkb (GEOMETRY and GEOGRAPHY have no Python mapping yet (P2-T181))")


def st_setsrid(geo: ColumnOrName, srid: Any) -> Column:
    """Not available: GEOMETRY and GEOGRAPHY have no Python mapping yet."""
    raise not_implemented("F.st_setsrid (GEOMETRY and GEOGRAPHY have no Python mapping yet (P2-T181))")


def st_srid(geo: ColumnOrName) -> Column:
    """Not available: GEOMETRY and GEOGRAPHY have no Python mapping yet."""
    raise not_implemented("F.st_srid (GEOMETRY and GEOGRAPHY have no Python mapping yet (P2-T181))")
