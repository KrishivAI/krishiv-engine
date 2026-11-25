# SPDX-License-Identifier: Apache-2.0
"""``Row``: a result row, as ``pyspark.sql.Row`` (a tuple with field names)."""

from __future__ import annotations

from typing import Any

from krishiv.errors import PySparkAttributeError, PySparkTypeError, PySparkValueError


class Row(tuple[Any, ...]):
    """``Row(a=1, b="x")``: access by position, ``row.a`` or ``row["a"]``."""

    __fields__: list[str]

    def __new__(cls, *args: Any, **kwargs: Any) -> Row:
        if args and kwargs:
            raise PySparkValueError("Can not use both args and kwargs to create Row")
        if kwargs:
            row = tuple.__new__(cls, kwargs.values())
            row.__fields__ = list(kwargs)
            return row
        return tuple.__new__(cls, args)

    def asDict(self, recursive: bool = False) -> dict[str, Any]:
        """The row as a dict; ``recursive`` also turns nested rows (in lists, dicts and rows) into dicts.

        With duplicate field names one of the duplicates is picked (the last), as PySpark.
        """
        if not hasattr(self, "__fields__"):
            raise PySparkTypeError(
                errorClass="CANNOT_CONVERT_TYPE",
                messageParameters={"from_type": "Row", "to_type": "dict"},
            )
        if recursive:
            return dict(zip(self.__fields__, (_conv(o) for o in self), strict=True))
        return dict(zip(self.__fields__, self, strict=True))

    def __call__(self, *args: Any) -> Row:
        """``Row("a", "b")(1, 2)``: a row class (names only) builds rows with those field names."""
        if len(args) > len(self):
            raise PySparkValueError(
                errorClass="TOO_MANY_VALUES",
                messageParameters={"expected": str(len(self)), "item": "fields", "actual": str(len(args))},
            )
        # As PySpark: fewer values name only the first fields (`Row("a", "b")(1)` is `Row(a=1)`).
        return _create_row(list(self)[: len(args)], list(args))

    def __contains__(self, item: object) -> bool:
        """Like PySpark: on a named row, ``in`` tests the field names."""
        if hasattr(self, "__fields__"):
            return item in self.__fields__
        return tuple.__contains__(self, item)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy rebuild through ``_create_row`` (``tuple``'s default would re-wrap the values)."""
        if hasattr(self, "__fields__"):
            return _create_row, (self.__fields__, list(self))
        return _create_row, (None, list(self))

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            try:
                return tuple.__getitem__(self, self.__fields__.index(item))
            except (AttributeError, ValueError):
                raise KeyError(item) from None
        return tuple.__getitem__(self, item)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise PySparkAttributeError(name) from None

    def __repr__(self) -> str:
        if hasattr(self, "__fields__"):
            return "Row(" + ", ".join(f"{k}={v!r}" for k, v in zip(self.__fields__, self, strict=True)) + ")"
        return f"<Row({', '.join(repr(v) for v in self)})>"


def _conv(obj: Any) -> Any:
    if isinstance(obj, Row):
        return obj.asDict(True)
    if isinstance(obj, list):
        return [_conv(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _conv(v) for k, v in obj.items()}
    return obj


def _create_row(fields: list[str] | None, values: list[Any]) -> Row:
    """Rebuilds a Row from its field names (``None`` for a positional row) and values."""
    if fields is None:
        return Row(*values)
    row = tuple.__new__(Row, values)
    row.__fields__ = list(fields)
    return row
