# SPDX-License-Identifier: Apache-2.0
"""Shared test setup."""

from __future__ import annotations

import os

# An unset session time zone is the machine's zone, so tests run in one zone on
# every machine (as `.cargo/config.toml` does for the Rust tests).
os.environ.setdefault("TZ", "UTC")
