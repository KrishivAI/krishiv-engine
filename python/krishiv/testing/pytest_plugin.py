# SPDX-License-Identifier: Apache-2.0
"""A pytest fixture for tests that run Spark code on Krishiv: ``krishiv_session``.

Enable it in a ``conftest.py``::

    pytest_plugins = ["krishiv.testing.pytest_plugin"]

or import it where it is used (``from krishiv.testing.pytest_plugin import krishiv_session``).
Nothing registers it by itself: installing ``krishiv`` does not change how any other
project's tests run.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from krishiv.sql.session import SparkSession


@pytest.fixture(scope="session")
def krishiv_session() -> Iterator[SparkSession]:
    """One in-process session for the whole test run, stopped when the run ends.

    The session is ``SparkSession.builder.getOrCreate()``: a test that sets a
    configuration key should restore it, or build its own session.
    """
    spark = SparkSession.builder.getOrCreate()
    try:
        yield spark
    finally:
        spark.stop()
