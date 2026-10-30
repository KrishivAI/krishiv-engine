# SPDX-License-Identifier: Apache-2.0
"""Engine metrics (``krishiv.metrics``): plain dicts from the active session."""

from __future__ import annotations

import krishiv


def test_metrics_lists_gauges_without_queries() -> None:
    series = krishiv.metrics()
    assert isinstance(series, list)
    by_name = {s["name"]: s for s in series}
    info = by_name["krishiv.build.info"]
    assert info["kind"] == "gauge"
    assert info["value"] == 1
    assert "version" in info["labels"]
    for s in series:
        assert {"name", "kind", "unit", "labels"} <= set(s)
        assert s["kind"] in ("counter", "gauge", "histogram")
        if s["kind"] == "histogram":
            assert isinstance(s["buckets"], list)
            assert "count" in s
            assert "sum" in s
        else:
            assert "value" in s


def test_metrics_counts_queries_when_enabled() -> None:
    spark = krishiv.session()
    spark.conf.set("krishiv.metrics.enabled", "true")
    try:
        before = {
            (s["name"], repr(sorted(s["labels"].items()))): s.get("value", s.get("count")) for s in krishiv.metrics()
        }
        spark.sql("SELECT 1").count()
        after = {
            (s["name"], repr(sorted(s["labels"].items()))): s.get("value", s.get("count")) for s in krishiv.metrics()
        }
        key = ("krishiv.query.count", repr(sorted({"status": "succeeded", "class": "interactive"}.items())))
        assert after.get(key, 0) == before.get(key, 0) + 1
    finally:
        spark.conf.unset("krishiv.metrics.enabled")
        spark.stop()
