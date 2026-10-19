# SPDX-License-Identifier: Apache-2.0
"""``Row``: a result row, as ``pyspark.sql.Row`` (a tuple with field names)."""

from __future__ import annotations

from typing import Any

from krishiv.errors import PySparkAttributeError, PySparkValueError


class Row(tuple):
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

    def asDict(self) -> dict[str, Any]:
        if not hasattr(self, "__fields__"):
            raise PySparkValueError("Cannot convert a Row class into dict")
        return dict(zip(self.__fields__, self, strict=True))

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
