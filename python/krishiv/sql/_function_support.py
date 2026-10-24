# SPDX-License-Identifier: Apache-2.0
"""What the generated function wrappers (``_generated_functions``) and ``functions`` call."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

from krishiv import _native
from krishiv.errors import PySparkValueError, not_expected_type
from krishiv.sql.column import Column, _literal, _named

__all__ = [
    "ColumnOrName",
    "_at_least",
    "_call",
    "_cols",
    "_deprecated",
    "_int_or_col",
    "_lit",
    "_named",
    "_num_or_col",
    "_one_of",
    "_opt",
]

ColumnOrName = Column | str


def _call(name: str, *args: _native.Column) -> Column:
    """A call of the registry function ``name``."""
    return Column(_native.call_function(name, list(args)))


def _num_or_col(value: Any, arg: str) -> _native.Column:
    """A Column or column name; an ``int`` or ``float`` is a literal (``atan2(1.0, "x")``), for the parameters
    PySpark annotates ``ColumnOrName | float``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return _literal(value)
    return _named(value, arg)


def _int_or_col(value: Any, arg: str) -> _native.Column:
    """A Column or column name; an ``int`` is a literal (``date_add("d", 1)``) and a ``float`` is not (it is
    neither a Column nor a name), for the parameters PySpark annotates ``ColumnOrName | int``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return _literal(value)
    return _named(value, arg)


def _lit(value: Any, arg: str) -> _native.Column:
    """A Column stays one; any other value is a literal (``lpad("s", 5, "-")``)."""
    if isinstance(value, Column):
        return value._c
    if value is None or isinstance(value, bool | int | float | str):
        return _literal(value)
    raise not_expected_type(arg, "Column, bool, int, float or str", value)


def _opt(convert: Callable[[Any, str], _native.Column], value: Any, arg: str) -> tuple[_native.Column, ...]:
    """The converted argument, or none when it is ``None`` (an optional argument left out)."""
    return () if value is None else (convert(value, arg),)


def _cols(values: Iterable[Any]) -> list[_native.Column]:
    return [_named(v, "cols") for v in values]


def _at_least(func_name: str, cols: tuple[Any, ...], n: int) -> None:
    if len(cols) < n:
        raise PySparkValueError(
            errorClass="WRONG_NUM_COLUMNS", messageParameters={"func_name": func_name, "num_cols": str(n)}
        )


def _one_of(arg: str, value: Any, allowed: tuple[int, ...]) -> None:
    if value not in allowed:
        raise PySparkValueError(
            errorClass="VALUE_NOT_ALLOWED",
            messageParameters={"arg_name": arg, "allowed_values": str(list(allowed))},
        )


def _deprecated(message: str) -> None:
    warnings.warn(message, FutureWarning, stacklevel=3)
