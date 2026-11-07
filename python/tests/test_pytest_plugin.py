# SPDX-License-Identifier: Apache-2.0
"""The ``krishiv_session`` fixture and the assertions it is for."""

from __future__ import annotations

import pytest

from krishiv.sql import Row
from krishiv.sql.session import SparkSession
from krishiv.testing import assertDataFrameEqual
from krishiv.testing.pytest_plugin import krishiv_session  # noqa: F401  (the fixture under test)


def test_the_fixture_is_a_working_session(krishiv_session: SparkSession) -> None:  # noqa: F811
    assert isinstance(krishiv_session, SparkSession)
    assert krishiv_session.sql("SELECT 1 AS a").collect() == [Row(a=1)]


def test_the_fixture_feeds_the_assertions(krishiv_session: SparkSession) -> None:  # noqa: F811
    got = krishiv_session.sql("SELECT * FROM VALUES (1, 'x'), (2, 'y') AS t(a, b)")
    assertDataFrameEqual(got, [Row(a=1, b="x"), Row(a=2, b="y")])
    with pytest.raises(AssertionError):
        assertDataFrameEqual(got, [Row(a=1, b="x"), Row(a=2, b="z")])
