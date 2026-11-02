# SPDX-License-Identifier: Apache-2.0
"""The offline profile viewer (``krishiv.ui``, ``ProfileView``) over a hand-built v1 document.

The document is shaped by ``docs/reference/query-profile-v1.schema.json``; no engine is needed for these
tests. The DataFrame entry points (``df.profile()``, ``explain(analyze=True)``) are covered at the end
against the built extension.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from krishiv.sql.profile_view import ProfileView, ui


def metrics(**over: int) -> dict[str, int]:
    base = {
        "drivers": 4,
        "rows_in": 1_000_000,
        "batches_in": 120,
        "rows_out": 250_000,
        "batches_out": 30,
        "compute_time_ns": 40_000_000,
        "max_driver_compute_time_ns": 12_000_000,
        "blocked_on_io_ns": 5_000_000,
        "max_driver_blocked_on_io_ns": 2_000_000,
        "blocked_on_memory_ns": 0,
        "blocked_on_sink_ns": 0,
        "morsels": 8,
        "peak_reserved_bytes": 3 * 1024 * 1024,
        "spills": 0,
        "spilled_bytes": 0,
    }
    base.update(over)
    return base


def profile() -> dict[str, Any]:
    return {
        "format": "krishiv.query-profile",
        "version": 1,
        "detailed": False,
        "query": {
            "wall_time_ns": 100_000_000,
            "workers": 1,
            "drivers": 8,
            "batches": 150,
            "yields": 3,
            "max_slice_ns": 4_000_000,
            "stalls": 0,
            "busy_ns": 90_000_000,
        },
        "pipelines": [
            {"id": 0, "drivers": 4, "threads": 4, "started_ns": 1_000_000, "ended_ns": 60_000_000},
            {"id": 1, "drivers": 4, "threads": 2, "started_ns": 60_000_000, "ended_ns": 100_000_000},
        ],
        "plan": [
            {"id": 0, "name": "HashAggregate", "detail": "keys=[l_returnflag]", "children": [1]},
            {"id": 1, "name": "Scan", "detail": "lineitem <script>alert(1)</script>", "children": []},
        ],
        "operators": [
            {
                "id": 0,
                "name": "FileScan",
                "node": 1,
                "role": "source",
                "pipelines": [0],
                "metrics": metrics(blocked_on_io_ns=30_000_000),
                "extra": [{"name": "files", "kind": "count", "value": 12}],
            },
            {
                "id": 1,
                "name": "HashAggregate",
                "node": 0,
                "role": "sink",
                "pipelines": [0, 1],
                "metrics": metrics(spills=2, spilled_bytes=2 * 1024 * 1024, blocked_on_memory_ns=1_000_000),
                "extra": [{"name": "index", "kind": "text", "value": "salted"}],
            },
            {
                "id": 2,
                "name": "Result",
                "node": None,
                "role": "sink",
                "pipelines": [1],
                "metrics": metrics(),
                "extra": [],
            },
        ],
    }


def test_the_page_shows_the_plan_the_operators_and_the_pipelines() -> None:
    html = ProfileView(profile()).to_html()
    assert html.startswith("<!doctype html>")
    for text in ("HashAggregate", "FileScan", "lineitem", "100.0 ms", "pipeline 0", "pipeline 1", "files 12"):
        assert text in html, text
    assert "spills 2 (2.0 MiB)" in html
    assert "index salted" in html
    assert "<h2>Result</h2>" in html
    # The scan is below the aggregate in the tree.
    assert html.index("HashAggregate") < html.index("lineitem")


def test_the_page_runs_no_script_and_escapes_the_document() -> None:
    html = ProfileView(profile()).to_html()
    assert "<script" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "http://" not in html
    assert "https://" not in html


def test_bars_are_scaled_to_the_heaviest_operator() -> None:
    doc = profile()
    html = ProfileView(doc).to_html()
    # The scan blocked 30 ms on I/O, the most of any figure: a full bar; its 40 ms compute is the maximum.
    assert "width:100.0%" in html
    assert "width:75.0%" in html  # 30 ms of 40 ms


def test_a_deep_plan_renders_without_recursion_trouble() -> None:
    doc = profile()
    plan = [{"id": i, "name": f"N{i}", "detail": "", "children": [i + 1] if i < 299 else []} for i in range(300)]
    doc["plan"] = plan
    doc["operators"] = [op for op in doc["operators"] if op["node"] is None]
    html = ProfileView(doc).to_html()
    assert "N299" in html


def test_a_cyclic_plan_does_not_loop() -> None:
    doc = profile()
    doc["plan"][1]["children"] = [0]
    assert "HashAggregate" in ProfileView(doc).to_html()


def test_unknown_members_are_ignored_and_missing_ones_rejected() -> None:
    doc = profile()
    doc["future_member"] = {"x": 1}
    doc["operators"][0]["extra"].append({"name": "later", "kind": "gauge", "value": 7})
    assert "FileScan" in ProfileView(doc).to_html()
    for member in ("query", "pipelines", "plan", "operators"):
        broken = copy.deepcopy(profile())
        del broken[member]
        with pytest.raises(ValueError, match=member):
            ProfileView(broken)
    with pytest.raises(ValueError, match=r"not a krishiv\.query-profile"):
        ProfileView({"format": "other"})
    newer = profile()
    newer["version"] = 2
    with pytest.raises(ValueError, match="version"):
        ProfileView(newer)


def test_json_text_a_file_and_the_notebook_fragment(tmp_path: Path) -> None:
    view = ProfileView(json.dumps(profile()))
    fragment = view._repr_html_()
    assert fragment.startswith("<style>")
    assert "<html" not in fragment
    out = view.save(tmp_path / "p.html")
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "wall=100.0 ms" in repr(view)


def test_ui_takes_documents_views_and_dataframes(tmp_path: Path) -> None:
    doc = profile()
    assert isinstance(ui(doc), ProfileView)
    view = ui(json.dumps(doc), path=tmp_path / "x.html")
    assert (tmp_path / "x.html").is_file()
    assert ui(view) is view

    json_text = json.dumps(doc)

    class FakeFrame:
        def explainString(self, *, analyze: bool, json: bool) -> str:  # PySpark's name
            assert analyze
            assert json
            return json_text

    assert "FileScan" in ui(FakeFrame())._repr_html_()
    with pytest.raises(TypeError, match="cannot view"):
        ui(42)


def test_dataframe_profile_and_explain_analyze() -> None:
    pytest.importorskip("krishiv._native")
    from krishiv.sql import SparkSession

    spark = SparkSession.builder.create()
    df = spark.sql("SELECT id % 3 AS k, count(*) AS n FROM range(1000) GROUP BY id % 3")
    view = df.profile()
    assert "Aggregate" in view._repr_html_()
    text = df.explainString(analyze=True)
    assert "Aggregate" in text
    assert json.loads(df.explainString(analyze=True, json=True))["format"] == "krishiv.query-profile"
