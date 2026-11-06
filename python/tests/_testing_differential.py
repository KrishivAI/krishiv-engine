# SPDX-License-Identifier: Apache-2.0
"""Shared runner for the ``krishiv.testing`` vs ``pyspark.testing`` differential (no imports of either library)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _type(types: Any, spec: Any) -> Any:
    if isinstance(spec, tuple):
        return types.ArrayType(_type(types, spec[1]), spec[2])
    return getattr(types, f"{spec}Type")()


def _schema(types: Any, fields: list[tuple]) -> Any:
    return types.StructType([types.StructField(n, _type(types, t), nullable) for n, t, nullable in fields])


def _rows(row_cls: Any, rows: list[dict]) -> list[Any]:
    return [row_cls(**r) for r in rows]


def run_scenario(
    kind: str, args: tuple, kwargs: dict, row_cls: Any, types: Any, assert_df: Callable, assert_schema: Callable
) -> None:
    actual, expected = args
    if kind == "rows":
        assert_df(_rows(row_cls, actual), _rows(row_cls, expected), **kwargs)
    else:
        assert_schema(_schema(types, actual), _schema(types, expected), **kwargs)


def outcome(fn: Callable[[], None]) -> dict[str, Any]:
    try:
        fn()
    except Exception as e:
        get = getattr(e, "getErrorClass", None) or getattr(e, "getCondition", None)
        return {"ok": False, "error_class": get() if get else type(e).__name__, "message": _ANSI.sub("", str(e))}
    return {"ok": True, "error_class": None, "message": None}
