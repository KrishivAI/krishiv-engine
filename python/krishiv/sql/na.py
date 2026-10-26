# SPDX-License-Identifier: Apache-2.0
"""``DataFrameNaFunctions`` (``DataFrame.na``): handling of missing data, as ``pyspark.sql.DataFrameNaFunctions``.

Everything here is built from the DataFrame facade (``filter``, ``select``, ``functions``), so the
results stay lazy and run in the engine. Columns are named by their top-level names (matched with the
session's case sensitivity; a backquoted name is unquoted). A name that is not one fails: with the
engine's analysis error if it resolves to nothing (or to several columns), and with
``KRISHIV_UNSUPPORTED_FEATURE`` if it is a qualified or nested reference, which these methods do not
handle yet. A DataFrame with duplicate column names (a join's result) works: its columns are renamed
by position while the expressions are built, and the result has the original names.

String columns are Arrow ``string_view`` columns in the tests: the engine cannot yet compare or
coalesce a ``Utf8`` column built from Python rows with a string literal (``Utf8`` against
``Utf8View``), so string ``fill`` and ``replace`` examples need ``string_view`` columns.
"""

from __future__ import annotations

import warnings
from functools import reduce
from typing import TYPE_CHECKING, Any

from krishiv.errors import PySparkTypeError, PySparkValueError, not_expected_type, unsupported_feature
from krishiv.sql import functions as F
from krishiv.sql.column import Column

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame

#: Marks an omitted ``value`` of ``replace``, where ``None`` is a value.
_NO_VALUE: Any = object()

_FLOATING = ("float", "double")
_INTEGRAL = ("tinyint", "smallint", "int", "bigint")


def _kind(value: Any) -> str | None:
    """The class of a literal: ``bool``, ``num`` or ``str`` (``None`` for anything else)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "num"
    if isinstance(value, str):
        return "str"
    return None


def _accepts(kind: str | None, dtype: str) -> bool:
    """Whether a column of type ``dtype`` takes part in a fill or replacement of values of class ``kind``:
    numbers go to every numeric column, strings to string columns and booleans to boolean columns."""
    if kind == "num":
        return dtype in _INTEGRAL or dtype in _FLOATING or dtype.startswith("decimal")
    if kind == "str":
        return dtype.startswith("string")
    return kind == "bool" and dtype == "boolean"


def _unquoted(name: str) -> str:
    """``name`` without the backquotes that quote it as a whole (a doubled backquote inside stands for one)."""
    if len(name) >= 2 and name[0] == "`" and name[-1] == "`":
        inner = name[1:-1]
        if "`" not in inner.replace("``", ""):
            return inner.replace("``", "`")
    return name


def _subset(subset: Any, expected: str) -> list[str]:
    """A ``subset`` argument as a list of names: a name, or a list or tuple of names."""
    if isinstance(subset, str):
        return [subset]
    if not isinstance(subset, list | tuple):
        raise not_expected_type("subset", expected, subset)
    for name in subset:
        if not isinstance(name, str):
            raise not_expected_type("subset", "str", name)
    return list(subset)


class _Schema:
    """The columns of a DataFrame, read once: the resolution of the names given against them, and the
    DataFrame to build expressions on (``base``, with unique column names ``refs``)."""

    def __init__(self, df: DataFrame) -> None:
        from krishiv.sql.dataframe import _case_sensitive, _has_duplicates

        self.df = df
        self.dtypes = df.dtypes
        self.names = [name for name, _ in self.dtypes]
        self.case_sensitive = _case_sensitive(df.sparkSession)
        self.renamed = _has_duplicates(self.names, self.case_sensitive)
        self.refs = [f"__krishiv_na_{i}" for i in range(len(self.names))] if self.renamed else self.names
        self.base = df.toDF(*self.refs) if self.renamed else df

    def index(self, name: str) -> int:
        """The position of the top-level column that ``name`` refers to. A name that resolves to nothing,
        or to several columns, fails with the engine's analysis error."""
        from krishiv.sql.dataframe import _same

        wanted = _unquoted(name)
        found = [i for i, column in enumerate(self.names) if _same(column, wanted, self.case_sensitive)]
        if len(found) == 1:
            return found[0]
        _ = self.df.select(F.col(name)).schema  # analysis: the engine's error for an unknown or ambiguous name
        raise unsupported_feature(f"DataFrame.na on the qualified or nested column reference {name}")

    def targets(self, subset: Any, expected: str) -> set[int] | None:
        """The positions of the columns in ``subset`` (``None``: all of them)."""
        if subset is None:
            return None
        return {self.index(name) for name in _subset(subset, expected)}

    def ref(self, i: int) -> Column:
        """The column at position ``i``, by its unique name."""
        from krishiv.sql.dataframe import _quoted

        return _quoted(self.refs[i])

    def project(self, changes: dict[int, Column]) -> DataFrame:
        """This DataFrame with the columns at the positions in ``changes`` replaced, in the same order."""
        if not changes:
            return self.df
        columns = []
        for i, name in enumerate(self.names):
            column = changes.get(i, self.ref(i))
            columns.append(column.alias(name) if i not in changes and self.renamed else column)
        return self.base.select(columns)

    def filter(self, condition: Column) -> DataFrame:
        """The rows of this DataFrame where ``condition`` (built from :meth:`ref`) holds."""
        kept = self.base.filter(condition)
        return kept.toDF(*self.names) if self.renamed else kept


def _present(ref: Column, dtype: str) -> Column:
    """Whether the column is neither null nor (for a floating-point column) NaN."""
    return ref.isNotNull() & ~ref.isNaN() if dtype in _FLOATING else ref.isNotNull()


def _filled(ref: Column, name: str, dtype: str, value: Any) -> Column:
    """The column with its nulls (and NaNs) replaced by ``value`` cast to the column's type."""
    if dtype in _FLOATING:
        # ``nanvl`` returns a double: cast back, so a float column stays a float column.
        ref = F.call_function("nanvl", ref, F.lit(None)).cast(dtype)
    return F.coalesce(ref, F.lit(value).cast(dtype)).alias(name)


def _replaced(ref: Column, name: str, dtype: str, mapping: dict[Any, Any]) -> Column:
    """The column with each key of ``mapping`` replaced by its value cast to the column's type."""
    # No float() detour: PySpark hands the JVM the ints as ints (``classic/dataframe.py`` passes
    # the ``rep_dict`` straight to ``_jmap``), so an int stays an int and keys past 2**53
    # still match exactly (``F.lit`` keeps them as 64-bit integers).

    (first, *rest) = [(ref == F.lit(old), F.lit(new).cast(dtype)) for old, new in mapping.items()]
    case = F.when(*first)
    for condition, new in rest:
        case = case.when(condition, new)
    return case.otherwise(ref).alias(name)


class DataFrameNaFunctions:
    """Functionality for working with missing data in a :class:`DataFrame`
    (``pyspark.sql.DataFrameNaFunctions``)."""

    def __init__(self, df: DataFrame) -> None:
        self.df = df

    def drop(self, how: str | None = "any", thresh: int | None = None, subset: Any = None) -> DataFrame:
        """Returns a new :class:`DataFrame` omitting rows with null or NaN values; see
        :meth:`DataFrame.dropna` (``df.na.drop`` and ``df.dropna`` are the same)."""
        if how is not None and (not isinstance(how, str) or how not in ("any", "all")):
            raise PySparkValueError(
                errorClass="VALUE_NOT_ANY_OR_ALL",
                messageParameters={"arg_name": "how", "arg_value": how},
            )
        if thresh is not None and not isinstance(thresh, int):
            raise not_expected_type("thresh", "int", thresh)
        schema = _Schema(self.df)
        if subset is None:
            positions = list(range(len(schema.names)))
        else:
            positions = [schema.index(name) for name in _subset(subset, "list, str or tuple")]
        if thresh is None:
            thresh = len(positions) if how == "any" else 1
        present = [_present(schema.ref(i), schema.dtypes[i][1]) for i in positions]
        if present and thresh == len(present):
            return schema.filter(reduce(lambda a, b: a & b, present))
        counts = [F.when(p, 1).otherwise(0) for p in present]
        total = reduce(lambda a, b: a + b, counts) if counts else F.lit(0)
        return schema.filter(total >= thresh)

    def fill(self, value: Any, subset: Any = None) -> DataFrame:
        """Returns a new :class:`DataFrame` with null (and, for floating-point columns, NaN) values
        replaced; see :meth:`DataFrame.fillna` (``df.na.fill`` and ``df.fillna`` are the same)."""
        if not isinstance(value, float | int | str | bool | dict):
            raise not_expected_type("value", "bool, dict, float, int or str", value)
        if isinstance(value, dict):
            return self._fill_each(value)
        if isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        schema = _Schema(self.df)
        targets = schema.targets(subset, "list or tuple")
        kind = _kind(value)
        return schema.project(
            {
                i: _filled(schema.ref(i), name, dtype, value)
                for i, (name, dtype) in enumerate(schema.dtypes)
                if (targets is None or i in targets) and _accepts(kind, dtype)
            }
        )

    def replace(self, to_replace: Any, value: Any = _NO_VALUE, subset: Any = None) -> DataFrame:
        """Returns a new :class:`DataFrame` replacing a value with another value; see
        :meth:`DataFrame.replace` (``df.na.replace`` and ``df.replace`` are the same)."""
        if value is _NO_VALUE:
            if not isinstance(to_replace, dict):
                raise PySparkTypeError(
                    errorClass="ARGUMENT_REQUIRED",
                    messageParameters={"arg_name": "value", "condition": "`to_replace` is dict"},
                )
            value = None
        sequences = (list, tuple)
        if not isinstance(to_replace, bool | float | int | str | list | tuple | dict):
            raise not_expected_type("to_replace", "bool, dict, float, int, str or tuple", to_replace)
        if (
            not isinstance(value, bool | float | int | str | list | tuple)
            and value is not None
            and not isinstance(to_replace, dict)
        ):
            raise not_expected_type("value", "bool, float, int, list, None, str or tuple", value)
        if isinstance(to_replace, sequences) and isinstance(value, sequences) and len(to_replace) != len(value):
            raise PySparkValueError(
                errorClass="LENGTH_SHOULD_BE_THE_SAME",
                messageParameters={
                    "arg1": "to_replace",
                    "arg2": "value",
                    "arg1_length": str(len(to_replace)),
                    "arg2_length": str(len(value)),
                },
            )
        # Like PySpark (``classic/dataframe.py`` checks lengths only when both are sequences,
        # then builds ``dict(zip(...))`` without ``strict``): a scalar ``to_replace`` with a
        # longer ``value`` list keeps the first pair instead of erroring.
        if subset is not None and not isinstance(subset, list | tuple | str):
            raise not_expected_type("subset", "list, str or tuple", subset)
        if isinstance(to_replace, bool | float | int | str):
            to_replace = [to_replace]
        if isinstance(to_replace, dict):
            mapping = to_replace
            if value is not None:
                warnings.warn("to_replace is a dict and value is not None. value will be ignored.", stacklevel=2)
        else:
            if isinstance(value, bool | float | int | str) or value is None:
                value = [value] * len(to_replace)
            mapping = dict(zip(to_replace, value, strict=False))
        kinds = {_kind(k) for k in mapping} | {_kind(v) for v in mapping.values() if v is not None}
        if len(kinds) > 1 or None in kinds:
            raise PySparkValueError(errorClass="MIXED_TYPE_REPLACEMENT", messageParameters={})
        schema = _Schema(self.df)
        targets = schema.targets(subset, "list, str or tuple")
        if not mapping:
            return self.df
        (kind,) = {_kind(k) for k in mapping}
        return schema.project(
            {
                i: _replaced(schema.ref(i), name, dtype, mapping)
                for i, (name, dtype) in enumerate(schema.dtypes)
                if (targets is None or i in targets) and _accepts(kind, dtype)
            }
        )

    def _fill_each(self, values: dict[str, Any]) -> DataFrame:
        """``fill`` with one replacement per column, each cast to its column's type (whatever that type is)."""
        for name, value in values.items():
            if not isinstance(name, str):
                raise not_expected_type("key type of dict", "str", name)
            if not isinstance(value, bool | int | float | str):
                raise not_expected_type("value type of dict", "bool, float, int or str", value)
        schema = _Schema(self.df)
        by_position = {schema.index(name): value for name, value in values.items()}
        return schema.project({i: _filled(schema.ref(i), *schema.dtypes[i], value) for i, value in by_position.items()})
