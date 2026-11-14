# SPDX-License-Identifier: Apache-2.0
"""Where the wrappers' Spark-facing names come from: the ``krishiv`` facade in-process, ``pyspark`` in a pool
worker.

The same wrapper code runs in two hosts (design ``P3-T014-connect-udfs-sandbox.md`` §3.1):

- **In-process** (the embedded runtime, P2-T059): the errors, types and rows are the ``krishiv`` facade's, and
  the runner conf comes from the native module (``_native._udf_runner_conf()``).
- **A pool worker** (the server's isolated workers, P3-T014): the interpreter has ``pyspark-client`` but no
  ``krishiv`` wheel. The server extracts this package under a stub ``krishiv/__init__.py`` that sets
  ``_UDF_WORKER = True``; the names then come from ``pyspark`` (a UDF's pickled return type is a
  ``pyspark.sql.types.DataType``), and the worker sets the runner conf of each batch with :func:`set_runner_conf`.

Every import of a ``krishiv.*`` module from ``krishiv._udf`` goes through here; nothing else in the package
may name ``krishiv.errors``, ``krishiv.sql`` or ``krishiv._native`` at module level.
"""

from __future__ import annotations

import sys
from typing import Any

#: True in a pool worker (the stub package's marker), False in-process.
IN_WORKER: bool = bool(getattr(sys.modules.get("krishiv"), "_UDF_WORKER", False))

if IN_WORKER:
    from pyspark.errors import (  # ty: ignore[unresolved-import]  # pool workers only
        PySparkImportError,
        PySparkNotImplementedError,
        PySparkRuntimeError,
        PySparkTypeError,
        PySparkValueError,
        UnsupportedOperationException,
    )

    def not_implemented(feature: str) -> PySparkNotImplementedError:
        """``krishiv.errors.not_implemented``: the same class and parameters, raised from ``pyspark``."""
        return PySparkNotImplementedError(errorClass="NOT_IMPLEMENTED", messageParameters={"feature": feature})

else:
    from krishiv.errors import (
        PySparkImportError,
        PySparkNotImplementedError,
        PySparkRuntimeError,
        PySparkTypeError,
        PySparkValueError,
        UnsupportedOperationException,
        not_implemented,
    )

__all__ = [
    "IN_WORKER",
    "PySparkImportError",
    "PySparkNotImplementedError",
    "PySparkRuntimeError",
    "PySparkTypeError",
    "PySparkValueError",
    "UnsupportedOperationException",
    "not_implemented",
    "runner_conf",
    "set_runner_conf",
]

_worker_conf: dict[str, str] = {}


def set_runner_conf(conf: dict[str, str]) -> None:
    """Sets the runner conf of the batch a pool worker is about to run (the worker only)."""
    global _worker_conf  # one conf per worker process, replaced per batch
    _worker_conf = conf


def runner_conf() -> dict[str, Any]:
    """The runner conf of the running call: the session's SQL conf subset Spark sends its worker, plus
    ``krishiv.python.udf.rows`` (the rows of a call without arguments)."""
    if IN_WORKER:
        return _worker_conf
    from krishiv import _native

    return _native._udf_runner_conf()
