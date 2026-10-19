# SPDX-License-Identifier: Apache-2.0
"""Krishiv: a Rust-native, Spark-compatible batch and streaming engine."""

from krishiv._native import __allocator__, __version__
from krishiv.sql.session import SparkSession

__all__ = ["SparkSession", "__version__", "build_info", "session"]


def build_info() -> dict[str, str]:
    """Return how this build was made, e.g. ``{"version": "0.1.0", "allocator": "mimalloc"}``."""
    return {"version": __version__, "allocator": __allocator__}


def session() -> SparkSession:
    """The active session, created on first use (``SparkSession.builder.getOrCreate()``)."""
    return SparkSession.builder.getOrCreate()
