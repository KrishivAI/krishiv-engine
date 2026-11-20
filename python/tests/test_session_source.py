# SPDX-License-Identifier: Apache-2.0
"""The client type an embedded session reports (P3-T003 §3: the ``source`` selector of a resource
group matches it)."""

from __future__ import annotations

from krishiv.sql import SparkSession


def test_an_embedded_session_reports_krishiv_python_as_its_client_type() -> None:
    spark = SparkSession.builder.create()
    rows = spark.sql("select client_type from system.sessions").collect()
    assert "krishiv-python" in [r[0] for r in rows]
