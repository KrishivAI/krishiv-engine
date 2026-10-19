# SPDX-License-Identifier: Apache-2.0
"""``Column``: a column expression, as ``pyspark.sql.Column`` (a thin wrapper on the native handle)."""

from __future__ import annotations

from typing import Any

from krishiv import _native
from krishiv.errors import PySparkTypeError, PySparkValueError, not_expected_type
from krishiv.sql.types import DataType, _simple_string_of


def _literal(value: Any) -> _native.Column:
    """The native column for a Python value used as a literal."""
    # bool is an int subclass; the native side tells them apart.
    if value is None or isinstance(value, bool | int | float | str):
        try:
            return _native.lit(value)
        except OverflowError as e:
            raise PySparkValueError(f"integer literal out of the 64-bit range: {value}") from e
    raise not_expected_type("value", "None, bool, int, float or str", value)


def _operand(value: Any) -> _native.Column:
    """A Column stays one; any other value is a literal (``col == "x"`` compares with a string)."""
    return value._c if isinstance(value, Column) else _literal(value)


def _named(value: Any, arg: str = "cols") -> _native.Column:
    """A Column stays one; a ``str`` names a column (``"*"`` is all of them)."""
    if isinstance(value, Column):
        return value._c
    if isinstance(value, str):
        return _native.col(value)
    raise not_expected_type(arg, "Column or str", value)


class Column:
    """A column expression. Build one with ``F.col``, ``df["name"]`` or an operator."""

    __slots__ = ("_c",)
    __hash__ = None  # __eq__ builds a Column, as in PySpark

    def __init__(self, native: _native.Column) -> None:
        self._c = native

    # Arithmetic.
    def __add__(self, other: Any) -> Column:
        return Column(self._c.add(_operand(other)))

    def __radd__(self, other: Any) -> Column:
        return Column(_operand(other).add(self._c))

    def __sub__(self, other: Any) -> Column:
        return Column(self._c.sub(_operand(other)))

    def __rsub__(self, other: Any) -> Column:
        return Column(_operand(other).sub(self._c))

    def __mul__(self, other: Any) -> Column:
        return Column(self._c.mul(_operand(other)))

    def __rmul__(self, other: Any) -> Column:
        return Column(_operand(other).mul(self._c))

    def __truediv__(self, other: Any) -> Column:
        return Column(self._c.div(_operand(other)))

    def __rtruediv__(self, other: Any) -> Column:
        return Column(_operand(other).div(self._c))

    def __mod__(self, other: Any) -> Column:
        return Column(self._c.rem(_operand(other)))

    def __rmod__(self, other: Any) -> Column:
        return Column(_operand(other).rem(self._c))

    def __neg__(self) -> Column:
        return Column(self._c.neg())

    # Comparison and logic.
    def __eq__(self, other: object) -> Column:  # ty: ignore[invalid-method-override]
        return Column(self._c.eq(_operand(other)))

    def __ne__(self, other: object) -> Column:  # ty: ignore[invalid-method-override]
        return Column(self._c.ne(_operand(other)))

    def __lt__(self, other: Any) -> Column:
        return Column(self._c.lt(_operand(other)))

    def __le__(self, other: Any) -> Column:
        return Column(self._c.le(_operand(other)))

    def __gt__(self, other: Any) -> Column:
        return Column(self._c.gt(_operand(other)))

    def __ge__(self, other: Any) -> Column:
        return Column(self._c.ge(_operand(other)))

    def __and__(self, other: Any) -> Column:
        return Column(self._c.and_(_operand(other)))

    def __rand__(self, other: Any) -> Column:
        return Column(_operand(other).and_(self._c))

    def __or__(self, other: Any) -> Column:
        return Column(self._c.or_(_operand(other)))

    def __ror__(self, other: Any) -> Column:
        return Column(_operand(other).or_(self._c))

    def __invert__(self) -> Column:
        return Column(self._c.invert())

    def __bool__(self) -> bool:
        raise PySparkValueError(errorClass="CANNOT_CONVERT_COLUMN_INTO_BOOL")

    def __iter__(self) -> Any:
        raise PySparkTypeError("Column is not iterable")

    def __repr__(self) -> str:
        return "Column<...>"

    def eqNullSafe(self, other: Any) -> Column:
        return Column(self._c.eq_null_safe(_operand(other)))

    def isNull(self) -> Column:
        return Column(self._c.is_null())

    def isNotNull(self) -> Column:
        return Column(self._c.is_not_null())

    def like(self, other: str) -> Column:
        if not isinstance(other, str):
            raise not_expected_type("other", "str", other)
        return Column(self._c.like(other))

    def isin(self, *cols: Any) -> Column:
        values = cols[0] if len(cols) == 1 and isinstance(cols[0], list | tuple | set) else cols
        return Column(self._c.isin([_operand(v) for v in values]))

    def between(self, lowerBound: Any, upperBound: Any) -> Column:
        return Column(self._c.between(_operand(lowerBound), _operand(upperBound)))

    def alias(self, *alias: str, **kwargs: Any) -> Column:
        if kwargs:
            raise PySparkValueError(f"unsupported alias argument(s): {sorted(kwargs)}")
        if len(alias) != 1:
            raise PySparkValueError("alias takes exactly one name")
        if not isinstance(alias[0], str):
            raise not_expected_type("alias", "str", alias[0])
        return Column(self._c.alias(alias[0]))

    name = alias

    def cast(self, dataType: DataType | str) -> Column:
        return Column(self._c.cast(_simple_string_of(dataType)))

    astype = cast

    # Sort orders: Spark's defaults are nulls first ascending, nulls last descending.
    def asc(self) -> Column:
        return Column(self._c.sort_order(True, True))

    def asc_nulls_first(self) -> Column:
        return self.asc()

    def asc_nulls_last(self) -> Column:
        return Column(self._c.sort_order(True, False))

    def desc(self) -> Column:
        return Column(self._c.sort_order(False, False))

    def desc_nulls_first(self) -> Column:
        return Column(self._c.sort_order(False, True))

    def desc_nulls_last(self) -> Column:
        return self.desc()
