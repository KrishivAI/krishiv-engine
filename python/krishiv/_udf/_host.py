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

import inspect
import sys
from typing import Any

#: True in a pool worker (the stub package's marker), False in-process.
IN_WORKER: bool = bool(getattr(sys.modules.get("krishiv"), "_UDF_WORKER", False))


def _spark_error(base: Any) -> Any:
    """The ``pyspark`` error class ``base`` as this package raises it: ``errorClass=`` and ``messageParameters=``.

    PySpark 3.5 names those arguments ``error_class`` and ``message_parameters``, and an older release lacks some
    of the 4.2.0 classes this package raises; such an error keeps its class and says the error class and its
    parameters. The subclass keeps ``base``'s name and module, so a traceback reads as the client expects.
    """
    snake = "error_class" in inspect.signature(base.__init__).parameters

    def __init__(
        self: Any,
        message: str | None = None,
        errorClass: str | None = None,
        messageParameters: dict[str, str] | None = None,
    ) -> None:
        if errorClass is None:
            base.__init__(self, message)
            return
        params = messageParameters or {}
        names = ("error_class", "message_parameters") if snake else ("errorClass", "messageParameters")
        try:
            base.__init__(self, **{names[0]: errorClass, names[1]: params})
        except Exception:  # the class or a parameter is not in this release's error classes
            base.__init__(self, f"[{errorClass}] {params}")

    body = {"__init__": __init__, "__module__": base.__module__, "__qualname__": base.__qualname__}
    return type(base.__name__, (base,), body)


if IN_WORKER:
    import pyspark.errors as _pyspark_errors  # ty: ignore[unresolved-import]  # pool workers only

    PySparkNotImplementedError = _spark_error(_pyspark_errors.PySparkNotImplementedError)
    PySparkRuntimeError = _spark_error(_pyspark_errors.PySparkRuntimeError)
    PySparkTypeError = _spark_error(_pyspark_errors.PySparkTypeError)
    PySparkValueError = _spark_error(_pyspark_errors.PySparkValueError)
    UnsupportedOperationException = _pyspark_errors.UnsupportedOperationException
    # PySpark 3.5 has no `PySparkImportError` (it is `PySparkException` and `ImportError`, as in 4.0).
    PySparkImportError = _spark_error(
        getattr(_pyspark_errors, "PySparkImportError", None)
        or type("PySparkImportError", (_pyspark_errors.PySparkException, ImportError), {})
    )

    def not_implemented(feature: str) -> Exception:
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
