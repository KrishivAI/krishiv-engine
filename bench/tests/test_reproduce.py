"""reproduce mode: worker protocol, process engines (timeout / crash), correctness verdicts, summaries."""

from __future__ import annotations

import contextlib
import io
import os
import select
import signal
import subprocess
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest

from krishiv_bench import reproduce, worker
from krishiv_bench.adapters import Dataset, Engine, Query, make_engine
from krishiv_bench.adapters.base import table_path
from krishiv_bench.adapters.process import EngineCrashed, EngineError, EngineTimeout, ProcessEngine


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Dataset:
    import duckdb

    root = tmp_path_factory.mktemp("tiny")
    duckdb.sql(f"COPY (SELECT range AS a, range % 3 AS g FROM range(1000)) TO '{root / 't.parquet'}' (FORMAT parquet)")
    return Dataset("tiny", 1, root, ["t"])


_BLOCKING_WORKER = """
import os, subprocess, sys, time
from krishiv_bench import worker

ready, pidfile = sys.argv[1], sys.argv[2]
out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
inp = sys.stdin.buffer
while True:
    req = worker.recv(inp)
    if req["op"] == "quit":
        break
    if req["op"] == "setup":
        worker.send(out, {"ok": {"version": "blocking", "config": {}, "build": {}}})
    elif req["query"].sql == "block":
        child = subprocess.Popen(["sleep", "300"])  # a descendant in the worker's process group
        open(pidfile, "w").write(str(child.pid))
        open(ready, "w").write("1")  # in flight: signal readiness, then block until killed
        time.sleep(3600)
    else:
        worker.send(out, {"ok": [(7,)] if req["op"] == "run" else (1, 0.001)})
"""


@pytest.fixture
def blocking_worker(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Make ProcessEngine start a test worker that blocks on the query ``block`` until it is killed.

    Returns (ready file, descendant pid file). No timing assumption: the worker signals that it is blocked, and
    the engine's timeout is the only thing that ends it.
    """
    script, ready, pidfile = tmp_path / "blocking_worker.py", tmp_path / "ready", tmp_path / "descendant.pid"
    script.write_text(_BLOCKING_WORKER)
    real_popen = subprocess.Popen

    def popen(argv, **kw):
        return real_popen([sys.executable, "-I", str(script), str(ready), str(pidfile)], **kw)

    monkeypatch.setattr(subprocess, "Popen", popen)
    return ready, pidfile


def q(sql: str, qid: str = "q01") -> Query:
    return Query(qid, sql, "tpch")


def test_frame_roundtrip_keeps_types():
    buf = io.BytesIO()
    worker.send(buf, {"ok": [(1, Decimal("1.50"), None, "x")]})
    buf.seek(0)
    assert worker.recv(buf) == {"ok": [(1, Decimal("1.50"), None, "x")]}


def test_recv_on_closed_pipe_raises():
    with pytest.raises(EOFError):
        worker.recv(io.BytesIO(b"\x01"))


def test_table_path_file_dir_and_missing(tmp_path: Path):
    (tmp_path / "a.parquet").write_bytes(b"")
    (tmp_path / "hits").mkdir()
    (tmp_path / "hits" / "p0.parquet").write_bytes(b"")
    ds = Dataset("d", 1, tmp_path, ["a", "hits"])
    assert table_path(ds, "a").endswith("a.parquet")
    assert table_path(ds, "hits").endswith("hits/*.parquet")
    with pytest.raises(FileNotFoundError):
        table_path(ds, "nope")


def test_make_engine_knows_the_reference_engines():
    for name in ("polars", "datafusion", "spark-local", "sail"):
        assert make_engine(name, threads=2, memory_limit="1g").name == name
    with pytest.raises(ValueError, match="unknown engine"):
        make_engine("nope")


def test_process_engine_runs_and_reports_errors(tiny):
    eng = ProcessEngine("duckdb", 2, 1, 60)
    eng.setup(tiny)
    try:
        assert eng.run(q("SELECT count(*), sum(a) FROM t")) == [(1000, 499500)]
        n, secs = eng.timed_run(q("SELECT g, count(*) FROM t GROUP BY g"))
        assert n == 3
        assert secs > 0
        with pytest.raises(EngineError, match="Catalog"):
            eng.run(q("SELECT * FROM missing"))
        assert eng.run(q("SELECT 1")) == [(1,)]  # still usable after an engine error
        assert eng.version()
        assert eng.config()["threads"] == 2
    finally:
        eng.teardown()


def test_process_engine_timeout_kills_and_restarts(tiny, blocking_worker):
    ready, _ = blocking_worker
    eng = ProcessEngine("duckdb", 2, 1, 1.0)
    eng.setup(tiny)
    try:
        with pytest.raises(EngineTimeout):
            eng.run(q("block"))
        assert ready.exists()  # the query was in flight (blocked), not finished
        assert eng._proc is None
        assert eng.run(q("SELECT 7")) == [(7,)]  # restarted lazily
    finally:
        eng.teardown()


def test_resolve_dataset_discovers_files_and_part_dirs(tmp_path: Path, monkeypatch):
    root = tmp_path / "clickbench"
    (root / "hits").mkdir(parents=True)
    (root / "hits" / "p0.parquet").write_bytes(b"")
    (root / "other.parquet").write_bytes(b"")
    (root / "empty").mkdir()
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    assert reproduce.resolve_dataset("clickbench", None, 1).tables == ["hits", "other"]
    with pytest.raises(SystemExit):
        reproduce.resolve_dataset("tpch", None, 1)


class Fake(Engine):
    """Engine stub returning canned rows."""

    def __init__(self, rows=None, exc: Exception | None = None):
        self.rows, self.exc = rows, exc

    def version(self) -> str:
        return "fake"

    def setup(self, dataset: Dataset) -> None:
        pass

    def run(self, query: Query):
        if self.exc:
            raise self.exc
        return self.rows


def test_correctness_flags_mismatch_error_and_timeout():
    qm = {n: q("SELECT x FROM t ORDER BY x") for n in ("duckdb", "krishiv", "a", "b", "c")}
    engines = {
        "duckdb": Fake([(1,), (2,)]),
        "krishiv": Fake([(1,), (2,)]),
        "a": Fake([(1,), (3,)]),
        "b": Fake(exc=EngineError("boom")),
        "c": Fake(exc=EngineTimeout("slow")),
    }
    verdict, nrows = reproduce.correctness(qm, engines)
    assert nrows == 2
    assert verdict["duckdb"]["status"] == verdict["krishiv"]["status"] == "ok"
    assert verdict["a"]["status"] == "mismatch"
    assert "value mismatch" in verdict["a"]["reason"]
    assert verdict["b"]["status"] == "error"
    assert verdict["c"]["status"] == "timeout"


def test_correctness_records_difference_from_krishiv_when_duckdb_agrees():
    qm = {n: q("SELECT x FROM t") for n in ("duckdb", "krishiv", "a")}
    engines = {"duckdb": Fake([(1,)]), "krishiv": Fake([(2,)]), "a": Fake([(1,)])}
    verdict, _ = reproduce.correctness(qm, engines)
    assert verdict["krishiv"]["status"] == "mismatch"
    assert verdict["a"]["status"] == "ok"
    assert "vs_krishiv" in verdict["a"]


def test_correctness_without_reference_marks_everything_failed():
    qm = {n: q("SELECT 1") for n in ("duckdb", "a")}
    verdict, nrows = reproduce.correctness(qm, {"duckdb": Fake(exc=EngineError("x")), "a": Fake([(1,)])})
    assert nrows == 0
    assert verdict["duckdb"]["status"] == "error"
    assert verdict["a"]["status"] == "error"


def test_ours_query_interleaves_and_summarizes(tiny):
    engines = {n: ProcessEngine(n, 2, 1, 60) for n in ("duckdb", "polars")}
    for e in engines.values():
        e.setup(tiny)
    try:
        qm = {n: q("SELECT g, count(*) AS c FROM t GROUP BY g") for n in engines}
        ent = reproduce.ours_query(qm, engines, reps=2, timeout_s=60)
    finally:
        for e in engines.values():
            e.teardown()
    assert ent["engines"]["duckdb"]["status"] == "ok"
    assert len(ent["engines"]["duckdb"]["samples_s"]) == 4
    pol = ent["engines"]["polars"]
    assert pol["status"] == "ok"
    assert len(pol["samples_s"]) == 4
    assert pol["ratio_ci95"][0] <= pol["ratio_ci95"][1]
    assert ent["engines"]["duckdb"]["cpu_seconds_median"] >= 0
    assert pol["cpu_seconds_median"] >= 0
    record = {
        "methods": ["ours"],
        "engines": {"duckdb": {}, "polars": {}},
        "queries": {"q01": ent, "q02": {"engines": {"polars": {"status": "mismatch"}}}},
        "load_at_start": 0.5,
    }
    s = reproduce.summarize(record)
    assert s["indicative"] is False
    assert s["ours"]["polars"]["n"] == 1
    assert s["ours"]["polars"]["statuses"] == {"ok": 1, "mismatch": 1}
    assert "polars" in reproduce.markdown(
        {**record, "suite": "tpch", "dataset": {"scale_factor": 1}, "pinned": {"threads": 2}, "summary": s}
    )


def test_process_engine_reports_the_cpu_seconds_of_a_timed_run(tiny):
    eng = ProcessEngine("duckdb", 2, 1, 120)
    eng.setup(tiny)
    try:
        assert eng.last_cpu_seconds is None
        eng.timed_run(q("SELECT sum(a * a) FROM range(300000000) t(a)"))
        busy = eng.last_cpu_seconds
        assert busy is not None
        assert busy > 0.05  # real work, read from the worker's /proc entries
        eng.timed_run(q("SELECT 1"))
        assert eng.last_cpu_seconds is not None
        assert eng.last_cpu_seconds < busy
    finally:
        eng.teardown()


def test_process_engine_has_no_cpu_reading_after_a_timeout(tiny, blocking_worker):
    eng = ProcessEngine("duckdb", 2, 1, 1.0)
    eng.setup(tiny)
    try:
        with pytest.raises(EngineTimeout):
            eng.timed_run(q("block"))
        assert eng.last_cpu_seconds is None
    finally:
        eng.teardown()


def test_theirs_query_best_of_runs(tiny):
    ok = reproduce.theirs_query("duckdb", q("SELECT count(*) FROM t"), tiny, 2, 1, 3, 60, 1)
    assert ok["status"] == "ok"
    assert len(ok["runs_s"]) == 3
    assert ok["best_s"] == min(ok["runs_s"])
    assert ok["first_s"] == ok["runs_s"][0]


def test_theirs_query_timeout(tiny, blocking_worker):
    ready, _ = blocking_worker
    slow = reproduce.theirs_query("duckdb", q("block"), tiny, 2, 1, 3, 1.0, 1)
    assert slow["status"] == "timeout"
    assert ready.exists()


def test_summarize_theirs_and_krishiv_gap():
    def eng(best):
        return {"status": "ok", "theirs": {"status": "ok", "best_s": best}}

    record = {
        "methods": ["theirs"],
        "engines": {"krishiv": {}, "duckdb": {}, "polars": {}},
        "queries": {
            "q01": {"engines": {"krishiv": eng(4.0), "duckdb": eng(1.0), "polars": eng(2.0)}},
            "q02": {"engines": {"krishiv": eng(9.0), "duckdb": eng(1.0), "polars": eng(1.0)}},
        },
        "load_at_start": 9.0,
    }
    s = reproduce.summarize(record)
    assert s["indicative"] is True
    assert s["theirs"]["krishiv"]["geomean"] == pytest.approx(6.0)
    assert s["theirs"]["polars"]["geomean"] == pytest.approx(2**0.5)
    assert s["theirs"]["polars"]["krishiv_over_engine_geomean"] == pytest.approx((2.0 * 9.0) ** 0.5)
    assert s["theirs"]["krishiv"]["worst_query"] == "q02"


def test_failed_setup_kills_worker_and_retries_clean(tmp_path: Path):
    ds = Dataset("d", 1, tmp_path, ["missing"])  # table_path raises in the worker's setup
    eng = ProcessEngine("duckdb", 2, 1, 60)
    with pytest.raises(EngineError):
        eng.setup(ds)
    assert eng._proc is None  # not left half-initialised; the next request starts a fresh worker
    with pytest.raises(EngineError):
        eng.run(q("SELECT 1"))
    assert eng._proc is None
    eng.teardown()


class BadSetup(Fake):
    def setup(self, dataset: Dataset) -> None:
        raise EngineError("no tables")


def test_setup_engines_drops_failed_non_reference_engine(tiny):
    engines = {"duckdb": Fake([(1,)]), "polars": BadSetup(), "datafusion": Fake([(1,)])}
    failed = reproduce.setup_engines(engines, tiny)
    assert list(engines) == ["duckdb", "datafusion"]
    assert failed["polars"]["status"] == "error"
    assert "no tables" in failed["polars"]["reason"]


def test_setup_engines_reference_failure_is_fatal(tiny):
    with pytest.raises(SystemExit, match="setup failed"):
        reproduce.setup_engines({"duckdb": BadSetup(), "polars": Fake([(1,)])}, tiny)


def test_cli_rejects_bad_methods(capsys):
    from krishiv_bench import cli

    for bad in ("our", "", "ours,,x"):
        with pytest.raises(SystemExit) as exc:
            cli.main(["reproduce", "--methods", bad])
        assert exc.value.code == 2
    assert "--methods" in capsys.readouterr().err


def _wait_for(path: Path) -> None:
    """Block until the worker signals readiness (an event, not a duration; the bound only stops a hung test)."""
    deadline = time.monotonic() + 60
    while not path.exists():
        assert time.monotonic() < deadline, "worker never signalled readiness"
        time.sleep(0.01)


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="needs Linux pidfd and /proc")
def test_kill_takes_down_descendants_in_the_worker_process_group(tiny, blocking_worker):
    """``ProcessEngine.kill`` is what the timeout path runs; it must reach the worker's descendants too."""
    ready, pidfile = blocking_worker
    eng = ProcessEngine("duckdb", 2, 1, None)  # no timeout: only the explicit kill below ends the worker
    eng.setup(tiny)
    errors: list[BaseException] = []

    def run_blocked():
        try:
            eng.run(q("block"))
        except EngineCrashed as e:  # the killed worker's pipe closes
            errors.append(e)

    t = threading.Thread(target=run_blocked)
    t.start()
    fd = -1
    try:
        _wait_for(ready)
        # The worker is alive and holds its child, so the pid is that child; a pidfd keeps naming it even if the
        # number is reused later.
        fd = os.pidfd_open(int(pidfile.read_text()))
        eng.kill()
        exited, _, _ = select.select([fd], [], [], 10)  # a pidfd becomes readable when its process exits
        assert exited, "descendant survived the process-group kill"
    finally:
        if fd >= 0:
            with contextlib.suppress(ProcessLookupError):
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            os.close(fd)
        eng.kill()
        t.join(30)
    assert len(errors) == 1


class CountingEngine(Fake):
    def __init__(self, counts):
        super().__init__()
        self.counts = iter(counts)

    def timed_run(self, query: Query):
        return next(self.counts), 0.1


def test_theirs_query_rejects_row_count_drift(tiny, monkeypatch):
    engines = iter([CountingEngine([3, 3]), CountingEngine([3, 4])])
    monkeypatch.setattr(reproduce, "ProcessEngine", lambda *a, **k: next(engines))
    ok = reproduce.theirs_query("x", q("SELECT 1"), tiny, 1, 1, 2, 60, nrows=3)
    assert ok["status"] == "ok"
    bad = reproduce.theirs_query("x", q("SELECT 1"), tiny, 1, 1, 2, 60, nrows=3)
    assert bad["status"] == "error"
    assert bad["reason"] == "EngineError: timed run returned 4 rows, expected 3"
    assert bad["error_class"] == "EngineError"
    assert bad["error_message"] == "timed run returned 4 rows, expected 3"
    assert "best_s" not in bad
    assert "runs_s" not in bad


def test_run_reproduce_rejects_nonpositive_reps_and_runs():
    with pytest.raises(SystemExit, match="at least 1"):
        reproduce.run_reproduce("tpch", 1, reps=0)
    with pytest.raises(SystemExit, match="at least 1"):
        reproduce.run_reproduce("tpch", 1, runs=-1)
