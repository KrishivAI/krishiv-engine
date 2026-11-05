"""Per-query resource telemetry, ``--cold`` page-cache eviction and the scale-ladder report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from krishiv_bench import ladder, resources
from krishiv_bench.adapters import Measure
from krishiv_bench.adapters.proc_mem import IoCounters, LocalProc, TreeProc, parse_io
from krishiv_bench.report import render

PROC_IO = """rchar: 1000
wchar: 5
syscr: 20
syscw: 1
read_bytes: 4096
write_bytes: 0
cancelled_write_bytes: 0
"""


def test_parse_io_reads_the_three_read_counters():
    assert parse_io(PROC_IO) == IoCounters(rchar=1000, read_bytes=4096, syscr=20)


def test_parse_io_refuses_text_without_the_counters():
    with pytest.raises(RuntimeError, match="syscr"):
        parse_io("rchar: 1\nread_bytes: 2\n")


def test_io_counters_subtract_and_add():
    a, b = IoCounters(10, 20, 3), IoCounters(4, 5, 1)
    assert a - b == IoCounters(6, 15, 2)
    assert (a - b) + b == a


def test_local_proc_reads_its_own_io():
    io = LocalProc("self").read_io()
    assert io.rchar > 0
    assert io.syscr > 0


def test_tree_window_counts_reads_and_cpu_of_a_child(tmp_path: Path):
    data = tmp_path / "blob"
    data.write_bytes(os.urandom(4 * 2**20))
    code = (
        "import sys, time\n"
        "print('ready', flush=True); sys.stdin.readline()\n"
        f"d = open({str(data)!r}, 'rb').read()\n"
        "t = time.process_time()\n"
        "while time.process_time() - t < 0.2: pass\n"
        "print('done', flush=True); sys.stdin.readline()\n"
    )
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdin is not None
    assert p.stdout is not None
    try:
        assert p.stdout.readline().strip() == "ready"
        tree = TreeProc(p.pid, interval_s=0.01)
        tree.start()
        p.stdin.write("\n")
        p.stdin.flush()
        assert p.stdout.readline().strip() == "done"
        tree.stop()
        assert tree.io is not None
        assert tree.io.rchar >= 4 * 2**20
        assert tree.io.syscr >= 1
        assert tree.cpu_seconds is not None
        assert tree.cpu_seconds >= 0.15
    finally:
        p.stdin.write("\n")
        p.stdin.close()
        p.wait(timeout=10)


def test_resource_fields_take_medians_and_leave_unknowns_none():
    meas = [
        Measure(peak_rss_bytes=10, cpu_seconds=4.0, rchar=1000, read_bytes=100, syscr=10),
        Measure(peak_rss_bytes=30, cpu_seconds=2.0, rchar=3000, read_bytes=300, syscr=10),
        Measure(peak_rss_bytes=20, cpu_seconds=6.0, rchar=2000, read_bytes=200, syscr=10),
    ]
    f = resources.resource_fields(meas, [2.0, 2.0, 2.0])
    assert f["peak_rss_bytes"] == 30
    assert f["cpu_seconds_median"] == 4.0
    assert f["avg_cores"] == 2.0
    assert f["read_bytes_median"] == 200
    assert f["rchar_median"] == 2000
    assert f["read_size_mean_bytes"] == 200
    assert f["spill_bytes"] is None


def test_resource_fields_without_any_measurement_are_all_none():
    f = resources.resource_fields([Measure(), Measure()], [1.0, 1.0])
    assert set(f.values()) == {None}


def test_evict_page_cache_advises_every_file_and_reports_it(tmp_path: Path, monkeypatch):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.parquet").write_bytes(b"x" * 100)
    (tmp_path / "sub" / "b.parquet").write_bytes(b"y" * 50)
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, off, ln, advice: calls.append((off, ln, advice)))
    assert resources.evict_page_cache(tmp_path) == {"files": 2, "bytes": 150}
    assert calls == [(0, 0, os.POSIX_FADV_DONTNEED)] * 2


def test_evict_page_cache_runs_for_real_and_skips_missing_roots(tmp_path: Path):
    (tmp_path / "a").write_bytes(b"z" * 10)
    assert resources.evict_page_cache(tmp_path)["files"] == 1
    assert resources.evict_page_cache(tmp_path / "missing") == {"files": 0, "bytes": 0}


def _result(engine: str, sf: float, p50: dict[str, float], cold: bool = False, started: str = "2026-11-05") -> dict:
    return {
        "suite": "tpch",
        "started_utc": started,
        "dataset": {"name": "tpch", "scale_factor": sf},
        "engine": {"name": engine},
        "cold": {"enabled": cold},
        "queries": {q: {"p50_s": t} for q, t in p50.items()},
    }


def _ladder(*rows: dict) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["dataset"]["scale_factor"], {})[r["engine"]["name"]] = r
    return out


def test_ladder_flags_only_ratio_growth_above_one_and_a_half():
    lad = _ladder(
        _result("duckdb", 10, {"q01": 1.0, "q06": 1.0, "q09": 1.0}),
        _result("krishiv", 10, {"q01": 1.3, "q06": 1.3, "q09": 1.3}),
        _result("duckdb", 100, {"q01": 10.0, "q06": 10.0, "q09": 10.0}),
        _result("krishiv", 100, {"q01": 14.0, "q06": 40.0, "q09": 19.0}),
    )
    got = {d["query"]: d for d in ladder.flagged(lad, "krishiv")}
    assert set(got) == {"q06"}  # q01 grows 1.08x, q09 1.46x, q06 2.36x
    assert got["q06"]["growth"] == pytest.approx(40 / 10 / 1.3)
    assert (got["q06"]["from_sf"], got["q06"]["to_sf"]) == (10, 100)


def test_ladder_compares_consecutive_scales_only():
    lad = _ladder(
        _result("duckdb", 1, {"q": 1.0}),
        _result("krishiv", 1, {"q": 1.0}),
        _result("duckdb", 10, {"q": 1.0}),
        _result("krishiv", 10, {"q": 1.4}),
        _result("duckdb", 100, {"q": 1.0}),
        _result("krishiv", 100, {"q": 1.9}),
    )
    assert ladder.flagged(lad, "krishiv") == []  # 1.4x then 1.36x: each step under 1.5x, though 1.9x overall


def test_ladder_render_warns_on_mixed_cold_and_warm():
    lad = _ladder(
        _result("duckdb", 10, {"q": 1.0}),
        _result("krishiv", 10, {"q": 1.0}),
        _result("duckdb", 100, {"q": 1.0}, cold=True),
        _result("krishiv", 100, {"q": 3.0}, cold=True),
    )
    md = ladder.render_ladder(lad)
    assert "cold and warm results are mixed" in md
    assert "| q | 10 | 100 |" in md


def test_ladder_needs_two_scales():
    with pytest.raises(SystemExit, match="two or more"):
        ladder.render_ladder(_ladder(_result("duckdb", 1, {"q": 1.0})))


def test_load_ladder_keeps_the_newest_result_per_scale_and_engine(tmp_path: Path):
    for i, (started, t) in enumerate([("2026-11-01", 5.0), ("2026-11-04", 2.0)]):
        (tmp_path / f"r{i}.json").write_text(json.dumps(_result("duckdb", 10, {"q": t}, started=started)))
    (tmp_path / "other.json").write_text(json.dumps({"suite": "tpcds", "dataset": {}, "engine": {"name": "x"}}))
    got = ladder.load_ladder("tpch", sorted(tmp_path.glob("*.json")))
    assert got[10.0]["duckdb"]["queries"]["q"]["p50_s"] == 2.0


def test_report_shows_the_resource_columns():
    q = {
        "p50_s": 1.0,
        "p99_s": 1.0,
        "rows_returned": 1,
        "cpu_seconds_median": 8.0,
        "avg_cores": 8.0,
        "read_bytes_median": 2**30,
        "rchar_median": 2 * 2**30,
        "read_size_mean_bytes": 262144.0,
    }
    r = {
        "suite": "tpch",
        "machine": {"cpu_model": "c", "cores_logical": 8, "ram_bytes": 2**34, "os": "linux"},
        "git": {"commit": "abc", "branch": "b", "dirty": False},
        "dataset": {"name": "tpch", "scale_factor": 1},
        "iterations": 1,
        "warmup": 0,
        "engine": {"name": "duckdb", "version": "1", "config": {}},
        "_file": "f.json",
        "queries": {"q01": q},
    }
    md = render({"duckdb": r})
    assert "max storage read GiB" in md
    assert "| 8.0 | 1.00 | 2.00 | 256 |" in md


# ---- wiring: run_suite, the CLI and the adapters


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    d = tmp_path / "tpch" / "sf1"
    d.mkdir(parents=True)
    (d / "t.parquet").write_bytes(b"")
    return tmp_path


def _run_fake_suite(monkeypatch: pytest.MonkeyPatch, cold: bool) -> tuple[dict, list[Path]]:
    from krishiv_bench import runner
    from krishiv_bench.adapters.base import Engine, Query

    class _Fake(Engine):
        def version(self) -> str:
            return "fake"

        def setup(self, dataset) -> None:
            pass

        def run(self, query):
            return [(1,)]

        def end_measure(self) -> Measure:
            return Measure(cpu_seconds=1.0, rchar=100, read_bytes=50, syscr=4)

    evicted: list[Path] = []
    monkeypatch.setattr(runner, "make_engine", lambda *a, **k: _Fake())
    monkeypatch.setattr(runner, "load_queries", lambda *a, **k: [Query("q01", "select 1", "tpch")])
    monkeypatch.setattr(runner, "evict_page_cache", lambda root: evicted.append(root))
    out = runner.run_suite("fake", "tpch", 1, iterations=2, warmup=1, cold=cold)
    return json.loads(out.read_text()), evicted


def test_run_suite_cold_evicts_before_every_run_and_records_it(data: Path, monkeypatch: pytest.MonkeyPatch):
    rec, evicted = _run_fake_suite(monkeypatch, cold=True)
    assert evicted == [data / "tpch" / "sf1"] * 3  # warm-up included
    assert rec["cold"]["enabled"] is True
    q = rec["queries"]["q01"]
    assert (q["rchar_median"], q["read_bytes_median"], q["syscr_median"]) == (100, 50, 4)
    assert q["read_size_mean_bytes"] == 25
    assert q["avg_cores"] > 0


def test_run_suite_warm_never_evicts(data: Path, monkeypatch: pytest.MonkeyPatch):
    rec, evicted = _run_fake_suite(monkeypatch, cold=False)
    assert evicted == []
    assert rec["cold"] == {"enabled": False}


def test_cli_passes_cold_and_dispatches_ladder(data: Path, monkeypatch: pytest.MonkeyPatch):
    from krishiv_bench import cli

    seen: dict = {}
    monkeypatch.setattr(cli, "run_suite", lambda *a, **k: seen.setdefault("args", a) and Path("/x"))
    cli.main(["run", "--engine", "duckdb", "--cold"])
    assert seen["args"][-1] is True
    monkeypatch.setattr(ladder, "write_ladder", lambda suite, files, dataset: Path("/ladder.md"))
    cli.main(["report", "--suite", "tpch", "--ladder"])


def test_krishiv_adapter_measures_its_runner_process(tmp_path: Path):
    from test_metrics import _FAKE_RUNNER

    from krishiv_bench.adapters import Dataset
    from krishiv_bench.adapters.krishiv import KrishivEngine

    fake = tmp_path / "runner"
    fake.write_text(_FAKE_RUNNER.replace("PYTHON", sys.executable))
    fake.chmod(0o755)
    eng = KrishivEngine(threads=1, binary=fake)
    eng.setup(Dataset("tpch", 1, tmp_path, ["t"]))
    try:
        eng.begin_measure()  # starts the runner
        m = eng.end_measure()
        assert m.peak_rss_bytes
        assert m.rchar is not None
        assert m.cpu_seconds is not None
    finally:
        eng.teardown()


def test_self_tree_measure_covers_the_harness_process():
    from krishiv_bench.adapters.proc_mem import SelfTreeMeasure

    class _E(SelfTreeMeasure):
        pass

    e = _E()
    e.begin_measure()
    sum(i * i for i in range(300_000))
    m = e.end_measure()
    assert m.peak_rss_bytes
    assert m.cpu_seconds is not None
    assert m.rchar is not None
    assert _E().end_measure() == Measure()  # no window: nothing measured
