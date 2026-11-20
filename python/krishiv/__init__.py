# SPDX-License-Identifier: Apache-2.0
"""Krishiv: a Rust-native, Spark-compatible batch and streaming engine."""

from typing import Any

from krishiv import _cpu

_cpu.ensure_supported()  # before the native module: an old CPU would die with SIGILL there

from krishiv._console import Console, ui  # noqa: E402
from krishiv._local import LocalConnect, start_local  # noqa: E402
from krishiv._native import __allocator__, __commit__, __version__  # noqa: E402
from krishiv._udf_hooks import register as _register_udf_hooks  # noqa: E402
from krishiv.sql.profile_view import ProfileView  # noqa: E402

_register_udf_hooks()
from krishiv._unsupported import SparkConf  # noqa: E402
from krishiv.sql.session import SparkSession  # noqa: E402

__all__ = [
    "Console",
    "LocalConnect",
    "ProfileView",
    "SparkConf",
    "SparkSession",
    "__version__",
    "build_info",
    "metrics",
    "session",
    "start_local",
    "ui",
]


def build_info() -> dict[str, str]:
    """Return how this build was made.

    ``commit`` is the git commit of the build, or ``"unknown"`` when the build was not
    stamped with one (a plain ``cargo build`` or a source archive). For example
    ``{"version": "0.2.1", "allocator": "mimalloc", "commit": "969250ca2aa8..."}``.
    """
    return {"version": __version__, "allocator": __allocator__, "commit": __commit__}


def session() -> SparkSession:
    """The active session, created on first use (``SparkSession.builder.getOrCreate()``)."""
    return SparkSession.builder.getOrCreate()


def metrics() -> list[dict[str, Any]]:
    """The engine's metrics snapshot as plain dicts (see ``Session.metrics``).

    Each dict has ``name``, ``kind`` (``counter``, ``gauge`` or ``histogram``),
    ``unit`` and ``labels``; counters and gauges add ``value`` (gauges also
    ``worker_id``); histograms add cumulative ``buckets`` as ``(le, count)``
    pairs with ``count`` and ``sum``. Query series appear only when
    ``krishiv.metrics.enabled`` is true; gauges are read when called.
    """
    return session()._native.metrics()
