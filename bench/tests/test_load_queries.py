"""load_queries refuses a missing queries directory instead of returning no queries (no engines, no timing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from krishiv_bench import paths, runner


@pytest.mark.parametrize("suite", ["tpcds", "tpch", "h2o-groupby", "h2o-join"])
def test_missing_queries_dir_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suite: str) -> None:
    # A non-editable install resolves bench_root() into site-packages, where no queries/ directory exists.
    monkeypatch.setattr(paths, "queries_dir", lambda s: tmp_path / "queries" / s)
    with pytest.raises(SystemExit, match=rf"{suite} queries directory .* is missing"):
        runner.load_queries(suite, "krishiv")


@pytest.mark.parametrize(("suite", "count"), [("tpch", 22), ("tpcds", 99)])
def test_present_queries_dir_loads(suite: str, count: int) -> None:
    assert len(runner.load_queries(suite, "krishiv")) == count


def test_present_h2o_queries_dir_loads() -> None:
    assert runner.load_queries("h2o-groupby", "krishiv")
