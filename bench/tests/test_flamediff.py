"""Differential flamegraph: folding, normalising, the diff format, share changes and the run (fake sampler)."""

from __future__ import annotations

from collections import Counter

import pytest

from krishiv_bench import flamediff
from krishiv_bench.adapters.base import Dataset, Query


def test_fold_reverses_frames_windows_samples_and_escapes_separators():
    samples = [
        (1.0, ["leaf", "mid", "root"]),
        (1.5, ["leaf", "mid", "root"]),
        (2.0, ["a;b", "root"]),  # t1 is exclusive
        (0.5, ["early", "root"]),  # before the window
        (1.2, []),  # no frames
    ]
    assert flamediff.fold(samples, 1.0, 2.0) == Counter({"root;mid;leaf": 2})
    assert flamediff.fold(samples, 1.0, 2.5) == Counter({"root;mid;leaf": 2, "root;a:b": 1})


def test_normalise_scales_to_the_total_and_handles_empty():
    assert flamediff.normalise(Counter({"a": 1, "b": 3}), 8) == {"a": 2, "b": 6}
    assert flamediff.normalise(Counter({"a": 1, "b": 99}), 10) == {"b": 10}  # whole counts; a zero is dropped
    assert flamediff.normalise(Counter(), 8) == {}


def test_diff_folded_covers_both_sides():
    rows = flamediff.diff_folded({"x": 1, "y": 2}, {"y": 3, "z": 4})
    assert rows == [("x", 1, 0), ("y", 2, 3), ("z", 0, 4)]


def test_shares_count_a_recursive_frame_once_per_stack():
    own, incl = flamediff.shares({"r;f;f;leaf": 3.0, "r;g": 1.0})
    assert own == {"leaf": 0.75, "g": 0.25}
    assert incl["f"] == 0.75
    assert incl["r"] == 1.0
    assert flamediff.shares({}) == ({}, {})


def test_top_changes_ranks_by_absolute_share_change_and_truncates():
    a, b = {"r;x": 8.0, "r;y": 2.0}, {"r;x": 4.0, "r;y": 6.0}
    ch = flamediff.top_changes(a, b, n=1)
    assert ch["self"] == [("x", 0.8, 0.4)]
    assert ch["inclusive"] == [("x", 0.8, 0.4)]
    full = flamediff.top_changes(a, b)
    assert {f for f, *_ in full["self"]} == {"x", "y"}


def test_render_summary_with_and_without_svg():
    ch = {"self": [("f|g", 0.1, 0.3)], "inclusive": []}
    with_svg = flamediff.render_summary("tpch", "q01", "a", "b", 10, 12, ch, True)
    assert "`diff.svg`" in with_svg
    assert "| `f/g` | 10.0% | 30.0% | +20.00 pp |" in with_svg
    without = flamediff.render_summary("tpch", "q01", "a", "b", 10, 12, ch, False)
    assert "not installed" in without


def test_render_svg_missing_tool_and_tool_failure(tmp_path, monkeypatch):
    diff = tmp_path / "d.folded"
    diff.write_text("a 1 2\n")
    monkeypatch.setattr(flamediff.shutil, "which", lambda n: None)
    assert flamediff.render_svg(diff, tmp_path / "o.svg", "t") is False
    fake = tmp_path / "fake-inferno"
    fake.write_text("#!/bin/sh\necho '<svg/>'\n")
    fake.chmod(0o755)
    (tmp_path / "o.svg").write_text("stale")
    monkeypatch.setattr(flamediff.shutil, "which", lambda n: None)
    assert flamediff.render_svg(diff, tmp_path / "o.svg", "t") is False
    assert not (tmp_path / "o.svg").exists()  # the chart of an earlier run is removed
    monkeypatch.setattr(flamediff.shutil, "which", lambda n: str(fake))
    assert flamediff.render_svg(diff, tmp_path / "o.svg", "t") is True
    assert (tmp_path / "o.svg").read_text() == "<svg/>\n"
    bad = tmp_path / "bad-inferno"
    bad.write_text("#!/bin/sh\necho oops >&2\nexit 2\n")
    bad.chmod(0o755)
    monkeypatch.setattr(flamediff.shutil, "which", lambda n: str(bad))
    assert flamediff.render_svg(diff, tmp_path / "o2.svg", "t") is False
    assert not (tmp_path / "o2.svg").exists()


def stub(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(flamediff, "resolve_dataset", lambda *a, **k: Dataset("d", 1.0, tmp_path, ["t"]))
    monkeypatch.setattr(flamediff, "load_queries", lambda s, e, only: [Query("q01", "select 1", s)])
    monkeypatch.setattr(flamediff.shutil, "which", lambda n: None)


def test_run_flamediff_writes_all_outputs(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    sides = {"A": Counter({"r;x": 8, "r;y": 2}), "B": Counter({"r;x": 2, "r;y": 3})}

    def sampler(runner, ds, q, runs, threads, mem, timeout):
        assert runs == 2
        return sides[runner]

    out = flamediff.run_flamediff("tpch", "q01", "A", "B", runs=2, out=tmp_path / "o", sampler=sampler)
    assert sorted(p.name for p in out.iterdir()) == ["a.folded", "b.folded", "diff.folded", "summary.md"]
    # B (5 samples) is scaled to A's 10: x 4, y 6.
    assert (out / "b.folded").read_text() == "r;x 4\nr;y 6\n"
    assert (out / "diff.folded").read_text() == "r;x 8 4\nr;y 2 6\n"
    assert "+40.00 pp" in (out / "summary.md").read_text()


def test_run_flamediff_names_the_failing_side(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)

    def sampler(runner, *a):
        if runner == "B":
            raise flamediff.RunnerCrashed("runner exited")
        return Counter({"r;x": 1})

    with pytest.raises(SystemExit, match=r"side B \(B\)"):
        flamediff.run_flamediff("tpch", "q01", "A", "B", sampler=sampler)


def test_run_flamediff_default_output_dir_and_errors(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    out = flamediff.run_flamediff("tpch", "q01", "A", "B", sampler=lambda *a: Counter({"r;x": 1}))
    assert out.parent == tmp_path / "data" / "results" / "flamediff"
    with pytest.raises(SystemExit, match="no perf samples for side B"):
        flamediff.run_flamediff(
            "tpch", "q01", "A", "B", sampler=lambda r, *a: Counter({"x": 1}) if r == "A" else Counter()
        )
    with pytest.raises(SystemExit, match="--runs"):
        flamediff.run_flamediff("tpch", "q01", "A", "B", runs=0)
    monkeypatch.setattr(flamediff, "load_queries", lambda *a: [])
    with pytest.raises(SystemExit, match="unknown query"):
        flamediff.run_flamediff("tpch", "q99", "A", "B")


def test_sample_side_folds_only_the_timed_window(tmp_path, monkeypatch):
    calls = []

    class Eng:
        def __init__(self, **k):
            self.launcher = k["launcher"]

        def setup(self, ds):
            calls.append("setup")

        def profile(self, q):
            calls.append("profile")

        def teardown(self):
            calls.append("teardown")

        def cancel(self):
            calls.append("cancel")

    monkeypatch.setattr(flamediff, "KrishivEngine", Eng)
    monkeypatch.setattr(flamediff, "perf_script", lambda data: "script")
    ticks = iter([10.0, 20.0])
    monkeypatch.setattr(flamediff.time, "clock_gettime", lambda c: next(ticks))

    def parse(text):
        assert text == "script"
        return [(15.0, ["leaf", "root"]), (25.0, ["late", "root"])]

    monkeypatch.setattr(flamediff, "parse_perf_script", parse)
    ds = Dataset("d", 1.0, tmp_path, ["t"])
    got = flamediff.sample_side("r", ds, Query("q01", "select 1", "tpch"), 2, 1, "1g", 5)
    assert got == Counter({"root;leaf": 1})
    assert calls == ["setup", "profile", "profile", "profile", "teardown", "cancel"]
