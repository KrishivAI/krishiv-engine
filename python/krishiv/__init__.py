# SPDX-License-Identifier: Apache-2.0
"""Krishiv: a Rust-native, Spark-compatible batch and streaming engine."""

from krishiv._native import __allocator__, __version__

__all__ = ["__version__", "build_info"]


def build_info() -> dict[str, str]:
    """Return how this build was made, e.g. ``{"version": "0.1.0", "allocator": "mimalloc"}``."""
    return {"version": __version__, "allocator": __allocator__}
