# SPDX-License-Identifier: Apache-2.0
"""``RuntimeConfig``: ``spark.conf``."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from krishiv import _native
from krishiv.errors import SparkNoSuchElementException, not_expected_type, not_implemented

if TYPE_CHECKING:
    from krishiv.sql.session import SparkSession

_UNSET = object()

# Settings that choose a cluster deployment: Krishiv runs in the process (or one `krishiv server`),
# so they are stored and read back but change nothing. A script written for a cluster must still
# start, so this warns (naming the key) and does not fail.
_CLUSTER_MASTER_PREFIXES = ("yarn", "k8s", "mesos", "spark://")
_CLUSTER_KEY_PREFIXES = ("spark.executor.", "spark.submit.", "spark.driver.", "spark.dynamicAllocation.")


def _warn_if_cluster_setting(key: str, value: str, stacklevel: int) -> None:
    """Warns that ``key=value`` has no effect when it chooses a cluster deployment. ``stacklevel`` counts
    from the caller of this function (1 is the caller), so the warning points at the user's line."""
    ignored = key.startswith(_CLUSTER_KEY_PREFIXES) or (
        key == "spark.master" and value.lower().startswith(_CLUSTER_MASTER_PREFIXES)
    )
    if ignored:
        warnings.warn(
            f"{key}={value} has no effect: Krishiv runs in this process (or in one `krishiv server`), "
            "not on a cluster manager.",
            UserWarning,
            stacklevel=stacklevel + 1,
        )


class RuntimeConfig:
    """``spark.conf``: get, set and unset the session's configuration."""

    def __init__(self, session: _native.Session, spark: SparkSession | None = None) -> None:
        self._session = session
        self._spark = spark

    def get(self, key: str, default: object = _UNSET) -> str | None:
        """The value of ``key``, or ``default`` when it is unset (fails when there is no default)."""
        value = self._session.conf_get(key)
        if value is not None:
            return value
        if default is _UNSET:
            # A key Spark registers and Krishiv does not answers Spark's default; an optional one, None (SV-3).
            found, spark_value = self._session.conf_get_as_spark(key)
            if found:
                return spark_value
            raise SparkNoSuchElementException(
                f'[SQL_CONF_NOT_FOUND] The SQL config "{key}" cannot be found. Please verify that the config exists. '
                "SQLSTATE: 42K0I",
                errorClass="SQL_CONF_NOT_FOUND",
                messageParameters={"sqlConf": f'"{key}"'},
                sqlState="42K0I",
            )
        return None if default is None else str(default)

    def set(self, key: str, value: str | int | bool) -> None:
        """Sets the configuration ``key`` for this session."""
        if not isinstance(value, str | int | bool):
            raise not_expected_type("value", "str, int or bool", value)
        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        # Spark refuses a core key such as `spark.executor.memory` at runtime (CANNOT_MODIFY_CONFIG): only a
        # setting that was taken warns.
        self._set(key, text)
        _warn_if_cluster_setting(key, text, stacklevel=2)

    def _set(self, key: str, text: str) -> None:
        self._session.conf_set(key, text)

    def unset(self, key: str) -> None:
        """Resets ``key`` to its default."""
        self._session.conf_unset(key)

    def isModifiable(self, key: str) -> bool:
        """Whether the configuration ``key`` is a known setting that can be changed at run time
        (``spark.conf.set``); static and unknown keys are ``False``, as in Spark."""
        if not isinstance(key, str):
            raise not_expected_type("key", "str", key)
        registered = self._session.conf_is_registered(key) or self._session.conf_is_spark_session_key(key)
        return registered and self._session.conf_is_modifiable(key)

    @property
    def getAll(self) -> dict[str, str]:
        """Every configuration set in this session, under the names the engine uses (a property, as in PySpark)."""
        if self._spark is None:
            raise not_implemented("RuntimeConfig.getAll on a configuration without a session")
        return {row[0]: row[1] for row in self._spark.sql("SET").collect()}
