"""Host-memory guard: MemAvailable wait, sampling, invalid-run retry in the runner, and the report section."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from krishiv_bench import memguard
from krishiv_bench.adapters import Measure
from krishiv_bench.report import invalid_sample_lines


def meminfo(path: Path, total_kb: int, avail_kb: int) -> Path:
    path.write_text(f"MemTotal:       {total_kb} kB\nMemFree:  1 kB\nMemAvailable:   {avail_kb} kB\nBuffers: 2 kB\n")
    return path


def test_read_meminfo_converts_kb_to_bytes(tmp_path):
    f = meminfo(tmp_path / "m", 1000, 250)
    assert memguard.read_meminfo(f) == (1000 * 1024, 250 * 1024)


def test_read_meminfo_without_memavailable_is_an_error(tmp_path):
    f = tmp_path / "m"
    f.write_text("MemTotal: 5 kB\n")
    with pytest.raises(ValueError, match="MemAvailable"):
        memguard.read_meminfo(f)


def test_threshold_is_a_fraction_of_total(tmp_path):
    f = meminfo(tmp_path / "m", 1000, 1000)
    assert memguard.threshold_bytes(0.75, f) == 750 * 1024


def test_wait_returns_at_once_when_memory_is_free(tmp_path):
    f = meminfo(tmp_path / "m", 1000, 900)
    reached, waited = memguard.wait_for_memory(750 * 1024, f, sleep=lambda s: pytest_fail())
    assert reached
    assert waited >= 0


def pytest_fail():
    raise AssertionError("must not sleep")


def test_wait_polls_until_memory_returns(tmp_path):
    f = meminfo(tmp_path / "m", 1000, 100)
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 3:
            meminfo(f, 1000, 800)

    reached, _ = memguard.wait_for_memory(750 * 1024, f, sleep=sleep)
    assert reached
    assert len(sleeps) == 3


def test_wait_gives_up_after_the_limit(tmp_path):
    f = meminfo(tmp_path / "m", 1000, 100)
    now = [0.0]

    def sleep(s):
        now[0] += s

    reached, waited = memguard.wait_for_memory(750 * 1024, f, poll_s=1, limit_s=5, sleep=sleep, clock=lambda: now[0])
    assert not reached
    assert waited >= 5


def test_sampler_records_the_lowest_value(tmp_path):
    f = meminfo(tmp_path / "m", 1000, 900)
    s = memguard.MemSampler(f, interval_s=0.01)
    with s:
        meminfo(f, 1000, 300)
        s._stop.wait(0.1)
        meminfo(f, 1000, 800)
        s._stop.wait(0.1)
    assert s.min_available == 300 * 1024


def test_is_invalid_only_below_the_threshold():
    assert memguard.is_invalid(100, 200)
    assert not memguard.is_invalid(200, 200)
    assert not memguard.is_invalid(None, 200)


def test_report_lists_invalid_samples_and_is_silent_without_them():
    inv = memguard.InvalidSample(3, 0, 2 * 1024**3, 8 * 1024**3).as_dict()
    results = {"krishiv": {"queries": {"q01": {"invalid_samples": [inv]}, "q02": {}}}}
    text = "\n".join(invalid_sample_lines(results, ["q01", "q02"]))
    assert "Invalid samples" in text
    assert "memory reclaimed" in text
    assert "| q01 | krishiv | 1 |" in text
    assert invalid_sample_lines({"krishiv": {"queries": {"q01": {}}}}, ["q01"]) == []


# ---- run_suite: invalid runs are retried and reported


def _suite(monkeypatch, tmp_path, avail_per_run, retries=2, warmup=0, guard=0.75, raises=()):
    """Run a fake one-query suite; ``avail_per_run[i]`` is the MemAvailable (kB) the i-th engine run sees.

    ``raises`` lists the engine-call indexes that raise after the dip (a query killed by reclaimed memory).
    """
    from krishiv_bench import runner
    from krishiv_bench.adapters.base import Engine, Query

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    d = tmp_path / "tpch" / "sf1"
    d.mkdir(parents=True)
    (d / "t.parquet").write_bytes(b"")
    info = tmp_path / "meminfo"
    meminfo(info, 1000, 900)
    calls = []

    class _Fake(Engine):
        def version(self):
            return "fake"

        def setup(self, dataset):
            pass

        def run(self, query):
            avail = avail_per_run[min(len(calls), len(avail_per_run) - 1)]
            calls.append(avail)
            meminfo(info, 1000, avail)
            memguard_sampler_tick()
            meminfo(info, 1000, 900)
            if len(calls) - 1 in raises:
                raise RuntimeError("killed")
            return [(1,)]

        def end_measure(self):
            return Measure(cpu_seconds=1.0)

    def memguard_sampler_tick():
        import time

        time.sleep(0.05)

    monkeypatch.setattr(memguard, "MEMINFO", info)
    real_sampler = memguard.MemSampler
    built = []

    def make_sampler():
        built.append(1)
        return real_sampler(info, 0.01)

    monkeypatch.setattr(memguard, "MemSampler", make_sampler)
    real_wait = memguard.wait_for_memory
    monkeypatch.setattr(memguard, "wait_for_memory", lambda t: real_wait(t, info, sleep=lambda s: None))
    monkeypatch.setattr(memguard, "threshold_bytes", lambda f: int(1000 * 1024 * f))
    monkeypatch.setattr(runner, "make_engine", lambda *a, **k: _Fake())
    monkeypatch.setattr(runner, "load_queries", lambda *a, **k: [Query("q01", "select 1", "tpch")])
    monkeypatch.setattr(runner, "dataset_provenance", lambda *a, **k: {})
    out = runner.run_suite("fake", "tpch", 1, iterations=2, warmup=warmup, min_mem_avail=guard, mem_retries=retries)
    rec = json.loads(out.read_text())
    rec["_samplers_built"] = len(built)
    return rec, calls


def test_run_suite_retries_a_run_that_dipped_and_records_it(tmp_path, monkeypatch):
    rec, calls = _suite(monkeypatch, tmp_path, [900, 100, 900, 900])
    q = rec["queries"]["q01"]
    assert len(calls) == 3  # two timed runs and one retry
    assert len(q["times_s"]) == 2
    assert [i["reason"] for i in q["invalid_samples"]] == ["memory reclaimed"]
    assert q["invalid_samples"][0]["min_mem_avail_bytes"] == 100 * 1024
    assert rec["mem_guard"]["retries"] == 2


def test_run_suite_fails_the_query_when_every_attempt_dips(tmp_path, monkeypatch):
    rec, calls = _suite(monkeypatch, tmp_path, [100], retries=1)
    q = rec["queries"]["q01"]
    assert "memory reclaimed" in q["error"]
    assert len(q["invalid_samples"]) == 2
    assert len(calls) == 2


def test_run_suite_fails_the_query_when_every_warmup_attempt_dips(tmp_path, monkeypatch):
    rec, calls = _suite(monkeypatch, tmp_path, [100], retries=1, warmup=1)
    q = rec["queries"]["q01"]
    assert "memory reclaimed" in q["error"]
    assert "times_s" not in q
    assert len(calls) == 2  # the warm-up's two attempts; no timed run follows


def test_run_suite_retries_a_query_that_raised_while_memory_was_reclaimed(tmp_path, monkeypatch):
    rec, calls = _suite(monkeypatch, tmp_path, [100, 900], raises={0})
    q = rec["queries"]["q01"]
    assert "error" not in q
    assert len(q["times_s"]) == 2
    assert [i["reason"] for i in q["invalid_samples"]] == ["memory reclaimed"]
    assert len(calls) == 3


def test_run_suite_still_fails_a_query_that_raised_with_memory_free(tmp_path, monkeypatch):
    rec, calls = _suite(monkeypatch, tmp_path, [900], raises={0})
    q = rec["queries"]["q01"]
    assert "killed" in q["error"]
    assert "invalid_samples" not in q
    assert len(calls) == 1


def test_run_suite_starts_no_sampler_when_the_guard_is_off(tmp_path, monkeypatch):
    rec, _ = _suite(monkeypatch, tmp_path, [100], guard=0)
    assert rec["_samplers_built"] == 0
    assert "invalid_samples" not in rec["queries"]["q01"]


def test_run_suite_clean_run_has_no_invalid_samples(tmp_path, monkeypatch):
    rec, _ = _suite(monkeypatch, tmp_path, [900])
    assert "invalid_samples" not in rec["queries"]["q01"]


def test_cli_exposes_the_guard_flags():
    from krishiv_bench import cli

    seen = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run_suite", lambda *a, **k: seen.update(k) or Path("/x"))
        cli.main(["run", "--engine", "duckdb", "--min-mem-avail", "0.5", "--mem-retries", "4"])
    assert (seen["min_mem_avail"], seen["mem_retries"]) == (0.5, 4)
