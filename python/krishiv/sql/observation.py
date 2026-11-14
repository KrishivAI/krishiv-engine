# SPDX-License-Identifier: Apache-2.0
"""``Observation``: named metrics of a DataFrame, collected as an action runs (``pyspark.sql.Observation``, 4.2).

The metrics are aggregate expressions given to :meth:`DataFrame.observe`; they are computed over the rows that
flow through that point of the plan while the first action runs, and :attr:`Observation.get` returns them as a
``dict`` keyed by the metric column names.
"""

from __future__ import annotations

from typing import Any

from krishiv import _native
from krishiv.errors import (
    PySparkAssertionError,
    PySparkException,
    PySparkValueError,
    not_expected_type,
)
from krishiv.sql import _convert

__all__ = ["Observation"]

# The engine's Krishiv-only classes and the PySpark ones the client raises for them.
_REUSED = "KRISHIV_OBSERVATION_REUSED"
_NO_OBSERVE = "KRISHIV_NO_OBSERVE_BEFORE_GET"


def _pyspark_error(error: PySparkException) -> PySparkException:
    """``error`` as PySpark's client raises it: the Krishiv-only classes become the PySpark ones."""
    condition = error.getCondition()
    if condition == _REUSED:
        return PySparkAssertionError(errorClass="REUSE_OBSERVATION", messageParameters={})
    if condition == _NO_OBSERVE:
        return PySparkAssertionError(errorClass="NO_OBSERVE_BEFORE_GET", messageParameters={})
    return error


def _non_empty(name: str) -> None:
    if name == "":
        raise PySparkValueError(
            errorClass="VALUE_NOT_NON_EMPTY_STR", messageParameters={"arg_name": "name", "arg_value": name}
        )


class Observation:
    """Metrics of one DataFrame, collected as its first action runs.

    One ``Observation`` observes one DataFrame. ``name`` is optional (a unique name is made up) and only keys the
    metrics inside the plan.

    >>> from krishiv.sql import Observation, functions as F
    >>> observation = Observation("stats")  # doctest: +SKIP
    >>> df.observe(observation, F.count(F.lit(1)).alias("rows")).count()  # doctest: +SKIP
    >>> observation.get  # doctest: +SKIP
    {'rows': 100}
    """

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            if not isinstance(name, str):
                raise not_expected_type("name", "str", name)
            _non_empty(name)
        try:
            self._native = _native.Observation(name)
        except PySparkException as e:  # an empty name only; checked above
            raise _pyspark_error(e) from e

    @property
    def name(self) -> str:
        """The name keying the metrics in the plan. A Krishiv extension: PySpark 4.2's ``Observation`` has only
        ``get``."""
        return self._native.name

    @property
    def get(self) -> dict[str, Any]:
        """The metrics as a ``dict`` keyed by the metric column names, once the first action on the observed
        DataFrame has finished (blocks until then, as Spark; later actions do not change it). An observed plan
        that never ran its metrics (``limit(0)`` above it) gives ``{}``."""
        try:
            stream = self._native.get()
        except PySparkException as e:
            raise _pyspark_error(e) from e
        pa = _convert.require_pyarrow("Observation.get")
        table = pa.table(stream)
        if table.num_columns == 0 or table.num_rows == 0:
            return {}
        return _convert.rows_of(table)[0].asDict()

    def __repr__(self) -> str:
        return f"Observation({self.name!r})"
