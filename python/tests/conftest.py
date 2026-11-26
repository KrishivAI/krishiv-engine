# SPDX-License-Identifier: Apache-2.0
"""Shared test setup."""

from __future__ import annotations

import gc
import os
import time

import pytest

# An unset session time zone is the machine's zone, so tests run in one zone on
# every machine (as `.cargo/config.toml` does for the Rust tests). `tzset` moves
# this interpreter's own conversions too; the engine reads the variable.
os.environ["TZ"] = "UTC"
time.tzset()


@pytest.fixture(autouse=True)
def _collect_sessions_after_each_test():
    """Free the sessions a test left behind.

    A session and its frames form reference cycles, and each owns tokio runtimes with worker threads.
    Without a collection the threads pile up over a few thousand tests and the macOS runners fail with
    "OS can't spawn worker thread" (os error 35).
    """
    yield
    gc.collect()
