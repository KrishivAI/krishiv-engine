"""load_queries refuses a missing queries directory instead of returning no queries (no engines, no timing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from krishiv_bench import runner


def test_missing_queries_dir_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A non-editable install resolves bench_root() into site-packages, where no queries/ directory exists.
    monkeypatch.setattr(runner, "queries_dir", lambda suite: tmp_path / "queries" / suite)
    with pytest.raises(SystemExit, match=r"tpcds queries directory .* is missing"):
        runner.load_queries("tpcds", "krishiv")


def test_present_queries_dir_loads() -> None:
    assert len(runner.load_queries("tpch", "krishiv")) == 22
