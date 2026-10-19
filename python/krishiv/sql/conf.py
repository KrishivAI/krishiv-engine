# SPDX-License-Identifier: Apache-2.0
"""``RuntimeConfig``: ``spark.conf``."""

from __future__ import annotations

from krishiv import _native
from krishiv.errors import PySparkValueError, not_expected_type

_UNSET = object()


class RuntimeConfig:
    def __init__(self, session: _native.Session) -> None:
        self._session = session

    def get(self, key: str, default: object = _UNSET) -> str | None:
        value = self._session.conf_get(key)
        if value is not None:
            return value
        if default is _UNSET:
            raise PySparkValueError(errorClass="CANNOT_BE_NONE", messageParameters={"arg_name": key})
        return None if default is None else str(default)

    def set(self, key: str, value: str | int | bool) -> None:
        if not isinstance(value, str | int | bool):
            raise not_expected_type("value", "str, int or bool", value)
        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        self._session.conf_set(key, text)

    def unset(self, key: str) -> None:
        self._session.conf_unset(key)
