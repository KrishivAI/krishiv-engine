# SPDX-License-Identifier: Apache-2.0
"""Krishiv: a Rust-native, Spark-compatible batch and streaming engine."""

import os

from krishiv import _cpu

_cpu.ensure_supported()  # before the native module: an old CPU would die with SIGILL there

# mimalloc reads its settings when the native module loads: purge freed memory after 10 ms and
# commit arena memory on demand, so the resident set stays close to what the memory pool tracks
# (krishiv_common::alloc::ALLOCATOR_ENV in the engine; values the user set win).
for _key, _value in (("MIMALLOC_PURGE_DELAY", "10"), ("MIMALLOC_ARENA_EAGER_COMMIT", "0")):
    os.environ.setdefault(_key, _value)

from krishiv._native import __allocator__, __version__  # noqa: E402
from krishiv.sql.session import SparkSession  # noqa: E402

__all__ = ["SparkSession", "__version__", "build_info", "session"]


def build_info() -> dict[str, str]:
    """Return how this build was made, e.g. ``{"version": "0.1.0", "allocator": "mimalloc"}``."""
    return {"version": __version__, "allocator": __allocator__}


def session() -> SparkSession:
    """The active session, created on first use (``SparkSession.builder.getOrCreate()``)."""
    return SparkSession.builder.getOrCreate()
