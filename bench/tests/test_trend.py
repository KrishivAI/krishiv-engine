"""Trend page: loading abba results, sparklines, changes and the rendered Markdown."""

from __future__ import annotations

import json
import math
from pathlib import Path

from krishiv_bench import trend


def write(directory: Path, name: str, when: str, ratios: dict[str, float | None], *, suite="tpch", sf=10, **extra):
    queries = {
        q: ({"status": "ok", "ratio": r} if r is not None else {"status": "failed", "reason": "x"})
        for q, r in ratios.items()
    }
    rec = {"kind": "abba", "suite": suite, "started_utc": when, "dataset": {"scale_factor": sf}, "queries": queries}
    rec.update(extra)
    (directory / name).write_text(json.dumps(rec))


def test_sparkline_scale_gaps_and_flat():
    assert trend.sparkline([]) == ""
    assert trend.sparkline([None, None]) == ""
    assert trend.sparkline([1.0, 1.0]) == "▅▅"
    line = trend.sparkline([1.0, None, 4.0, 2.0])
    assert line[0] == "▁"
    assert line[1] == " "
    assert line[2] == "█"
    assert line[3] == "▅" or line[3] in trend.BARS


def test_change_flat_up_down_and_missing():
    assert trend.change([1.0, 1.02]) == "="
    assert trend.change([1.0, 1.5]) == "+50%"
    assert trend.change([2.0, 1.0]) == "-50%"
    assert trend.change([None, 1.0]) == "-"
    assert trend.change([1.0, None]) == "-"
    assert trend.change([1.0, None, 3.0]) == "+200%"


def test_load_runs_groups_sorts_and_ignores_junk(tmp_path):
    write(tmp_path, "2_abba_tpch_sf10.json", "2026-10-02T00:00:00", {"q1": 1.5, "q2": None})
    write(tmp_path, "1_abba_tpch_sf10.json", "2026-10-01T00:00:00", {"q1": 2.0, "q2": 1.0}, summary={"geomean": 1.4})
    write(tmp_path, "3_abba_tpcds_sf1.json", "2026-10-03T00:00:00", {"q1": 1.0}, suite="tpcds", sf=1.0)
    write(tmp_path, "4_abba_tpch_sf10.json", "2026-10-04T00:00:00", {"q1": None})  # nothing passed
    (tmp_path / "5_abba_bad.json").write_text("{not json")
    (tmp_path / "6_abba_odd.json").write_text(json.dumps({"kind": "abba", "queries": []}))
    (tmp_path / "7_abba_other.json").write_text(json.dumps({"kind": "ab"}))
    (tmp_path / "8_abba_list.json").write_text("[]")
    groups = trend.load_runs(tmp_path)
    assert sorted(groups) == [("tpcds", "1"), ("tpch", "10")]
    runs = groups[("tpch", "10")]
    assert [r["when"][:10] for r in runs] == ["2026-10-01", "2026-10-02"]
    assert runs[0]["geomean"] == 1.4
    assert math.isclose(runs[1]["geomean"], 1.5)  # computed from the passing queries
    assert runs[1]["ratios"]["q2"] is None


def test_render_marks_failures_indicative_runs_and_limits_columns(tmp_path):
    write(tmp_path, "a_abba_x.json", "2026-10-01T00:00:00", {"q1": 2.0, "q10": 1.0})
    write(tmp_path, "b_abba_x.json", "2026-10-02T00:00:00", {"q1": 1.0, "q10": None}, summary={"indicative": True})
    write(tmp_path, "c_abba_x.json", "2026-10-03T00:00:00", {"q1": 1.0, "q2": 3.0})
    text = trend.render(trend.load_runs(tmp_path), last=2)
    assert "2026-10-01" not in text
    assert "| query | 2026-10-02 | 2026-10-03 | trend | change |" in text
    assert "| q1 | 1.00* | 1.00 |" in text
    assert "| q10 | x |  |" in text  # failed, then absent
    assert text.index("| q1 ") < text.index("| q2 ") < text.index("| q10 ")  # natural query order
    assert "**geomean**" in text


def test_render_without_results():
    assert "No `abba` results found." in trend.render({})


def test_run_trend_default_and_custom_output(tmp_path, monkeypatch, capsys):
    write(tmp_path, "a_abba_x.json", "2026-10-01T00:00:00", {"q1": 2.0})
    out = trend.run_trend(directory=tmp_path)
    assert out == tmp_path / "trend.md"
    assert "| q1 | 2.00 |" in out.read_text()
    custom = trend.run_trend(tmp_path / "docs" / "t.md", 5, tmp_path)
    assert custom.is_file()
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    assert trend.run_trend().parent == tmp_path / "results"
    import pytest

    with pytest.raises(SystemExit, match="--last"):
        trend.run_trend(last=0, directory=tmp_path)


def test_cli_trend_bisect_and_flamediff_dispatch(tmp_path, monkeypatch):
    from krishiv_bench import bisect, cli, flamediff

    write(tmp_path, "a_abba_x.json", "2026-10-01T00:00:00", {"q1": 2.0})
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path.parent / "nodata"))
    cli.main(["trend", "--out", str(tmp_path / "o.md"), "--last", "3"])
    assert (tmp_path / "o.md").is_file()

    seen = {}
    monkeypatch.setattr(bisect, "run_bisect", lambda *a: seen.update(bisect=a))
    cli.main(["bisect", "--query", "q07", "--good", "g", "--bad", "b", "--time", "--threshold", "1.2"])
    assert seen["bisect"][:5] == ("tpch", "q07", "g", "b", "time")
    assert seen["bisect"][-1] == 1.2
    cli.main(["bisect", "--query", "q07", "--good", "g", "--bad", "b"])
    assert seen["bisect"][4] == "correctness"

    def boom(*a):
        raise bisect.BisectError("nope")

    monkeypatch.setattr(bisect, "run_bisect", boom)
    import pytest

    with pytest.raises(SystemExit, match="bisect: nope"):
        cli.main(["bisect", "--query", "q07", "--good", "g", "--bad", "b"])

    monkeypatch.setattr(flamediff, "run_flamediff", lambda *a: seen.update(fd=a))
    cli.main(["flamediff", "--query", "q01", "--a-runner", "A", "--b-runner", "B", "--out", "/x"])
    assert seen["fd"][:4] == ("tpch", "q01", "A", "B")
    assert str(seen["fd"][-1]) == "/x"
