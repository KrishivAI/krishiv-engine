"""ABBA statistics, the report, and the Krishiv adapter against a fake runner speaking the line protocol."""

from __future__ import annotations

import json
import math
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from krishiv_bench import abba
from krishiv_bench.adapters.base import Dataset, Engine, Query
from krishiv_bench.adapters.krishiv import KrishivEngine, RunnerCrashed, RunnerError

FAKE_RUNNER = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import sys
    from urllib.parse import unquote
    import pyarrow as pa, pyarrow.parquet as pq, os
    print("READY 9.9.9 t", flush=True)
    for line in sys.stdin:
        cmd = line.split()
        if cmd[0] == "QUIT":
            break
        sql = open(unquote(cmd[1])).read()
        if "boom" in sql:
            print("ERR planning failed", flush=True)
        elif "hang" in sql:
            import time; time.sleep(60)
        elif "die" in sql:
            sys.stderr.write("fatal: out of memory\\n"); sys.stderr.flush()
            os._exit(1)
        elif "garbage" in sql:
            print("what", flush=True)
        elif cmd[0] == "RUN":
            print("OK 2 0.125000", flush=True)
        else:
            out = unquote(cmd[2]); os.makedirs(out)
            pq.write_table(pa.table({{"a": [1, 2], "b": ["x", "y"]}}), out + "/part-0.parquet")
            print("OK 2 0.125000", flush=True)
    """
)


@pytest.fixture
def engine(tmp_path: Path):
    binary = tmp_path / "runner"
    binary.write_text(FAKE_RUNNER)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    eng = KrishivEngine(threads=2, memory_limit="1g", binary=binary)
    eng.timeout_s = 30  # generous: process start-up on a loaded host must not fail the ok paths
    eng.setup(Dataset("tpch", 1, tmp_path, ["t"]))
    yield eng
    eng.teardown()


def q(sql: str) -> Query:
    return Query("q99", sql, "tpch")


def test_adapter_run_and_timed_run(engine: KrishivEngine):
    assert engine.version() == "9.9.9"
    assert engine.timed_run(q("select 1")) == (2, 0.125)
    assert engine.run(q("select 1")) == [(1, "x"), (2, "y")]
    assert engine.config()["threads"] == 2
    assert engine.build_info()["allocator"] == "mimalloc"


def test_adapter_error_crash_and_restart(engine: KrishivEngine):
    with pytest.raises(RunnerError, match="planning failed"):
        engine.timed_run(q("boom"))
    assert engine.timed_run(q("ok"))[0] == 2  # an ERR keeps the process serving
    with pytest.raises(RunnerCrashed):
        engine.timed_run(q("die"))
    assert engine.timed_run(q("ok"))[0] == 2  # restarted on the next command
    with pytest.raises(RunnerCrashed, match="unexpected reply"):
        engine.timed_run(q("garbage"))
    assert engine._proc is None  # killed: the next command starts a fresh runner
    engine.timeout_s = 0.5
    with pytest.raises(RunnerCrashed, match="no answer"):
        engine.timed_run(q("hang"))
    assert engine.timed_run(q("ok"))[0] == 2


def test_missing_binary(tmp_path: Path):
    eng = KrishivEngine(binary=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        eng.setup(Dataset("tpch", 1, tmp_path, ["t"]))


def test_bootstrap_and_geomean():
    a, b = [2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]
    assert abba.bootstrap_ci(a, b) == (2.0, 2.0)
    assert math.isclose(abba.geomean([1.0, 4.0]), 2.0)
    lo, hi = abba.geomean_ci([(a, b), ([8.0] * 4, b)])
    assert lo == hi == pytest.approx(4.0)


def _record(load: float, ratio_ok: bool = True) -> dict:
    ok = {
        "status": "ok",
        "krishiv_s": [0.2] * 4,
        "duckdb_s": [0.1] * 4,
        "krishiv_median_s": 0.2,
        "duckdb_median_s": 0.1,
        "ratio": 2.0,
        "ratio_ci95": [1.9, 2.1],
        "load_before": load,
        "load_after": load,
    }
    return {
        "load_at_start": load,
        "queries": {"q01": ok, "q02": {"status": "failed", "reason": "wrong result: a|b"}},
    }


def test_summary_and_markdown_flag_load_and_failures():
    rec = _record(9.0)
    rec["summary"] = abba.summarize(rec)
    s = rec["summary"]
    assert s["indicative"]
    assert s["queries_ok"] == 1
    assert "q02" in s["failed"]
    assert s["geomean"] == pytest.approx(2.0)
    assert s["worst_query"] == "q01"
    text = abba.markdown(rec)
    assert "INDICATIVE ONLY" in text
    assert "| q02 | failed |" in text
    assert "2.00x" in text
    quiet = _record(1.0)
    assert not abba.summarize(quiet)["indicative"]
    none = {"load_at_start": 1.0, "queries": {"q01": {"status": "failed", "reason": "x"}}}
    none["summary"] = abba.summarize(none)
    assert "geomean" not in none["summary"]
    assert "No query passed" in abba.markdown(none)


class _Duck(Engine):
    """DuckDB stand-in: fixed rows, fixed time."""

    def __init__(self, rows):
        self.rows = rows

    def version(self):
        return "stub"

    def setup(self, dataset):
        del dataset

    def run(self, query):
        del query
        return self.rows

    def cancel(self):
        pass


def test_one_query_paths(engine: KrishivEngine):
    duck = _Duck([(1, "x"), (2, "y")])
    done = abba._one_query(q("select 1"), engine, duck, 1, 5)
    assert done["status"] == "ok"
    assert len(done["krishiv_s"]) == 2
    assert done["krishiv_s"][0] == 0.125
    wrong = abba._one_query(q("select 1"), engine, _Duck([(9, "z"), (2, "y")]), 1, 5)
    assert wrong["status"] == "failed"
    assert "wrong result" in wrong["reason"]
    err = abba._one_query(q("boom"), engine, duck, 1, 5)
    assert "krishiv error" in err["reason"]
    crash = abba._one_query(q("die"), engine, duck, 1, 5)
    assert "crashed" in crash["reason"]

    class Broken(_Duck):
        def run(self, query):
            raise RuntimeError("duck down")

    assert "duckdb RuntimeError" in abba._one_query(q("x"), engine, Broken([]), 1, 5)["reason"]
    short = abba._one_query(q("select 1"), engine, _Duck([(1, "x"), (2, "y"), (3, "z")]), 1, 5)
    assert short["status"] == "failed"  # 2 rows vs 3: caught by the correctness check before timing


def test_run_abba_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = tmp_path / "data"
    ds = data / "tpch-duckdb" / "sf1"
    ds.mkdir(parents=True)
    pq.write_table(pa.table({"a": [1, 2], "b": ["x", "y"]}), ds / "t.parquet")
    binary = tmp_path / "runner"
    binary.write_text(FAKE_RUNNER)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("KRISHIV_DATA", str(data))
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(binary))
    monkeypatch.setattr(abba, "load_queries", lambda *_: [Query("q01", "select a, b from t order by a", "tpch")])
    events: list[str] = []
    monkeypatch.setattr(abba, "wait_idle", lambda: events.append("wait"))
    real_setup = KrishivEngine.setup

    def recording_setup(self, dataset):
        events.append("start")
        real_setup(self, dataset)

    monkeypatch.setattr(KrishivEngine, "setup", recording_setup)
    out = abba.run_abba(1, reps=2, threads=2, memory_limit="1g", timeout_s=30)
    assert events == ["wait", "start"]
    result = json.loads(out.read_text())
    assert result["queries"]["q01"]["status"] == "ok"
    assert len(result["queries"]["q01"]["duckdb_s"]) == 4
    assert result["summary"]["queries_ok"] == 1
    assert "Geomean over 1/1" in capsys.readouterr().out
    assert abba.report_from_file(out).startswith("| query")
    with pytest.raises(SystemExit):
        abba.run_abba(7)


def test_crash_message_carries_runner_stderr(engine: KrishivEngine):
    with pytest.raises(RunnerCrashed, match="fatal: out of memory"):
        engine.timed_run(q("die"))


def test_startup_failure_reports_stderr(tmp_path: Path):
    binary = tmp_path / "bad"
    binary.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write('krishiv-bench-runner: no tables\\n')\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    eng = KrishivEngine(binary=binary)
    with pytest.raises(RunnerCrashed, match="no tables"):
        eng.setup(Dataset("tpch", 1, tmp_path, ["t"]))
    eng.teardown()


def test_paths_with_spaces_are_encoded(tmp_path: Path):
    binary = tmp_path / "run ner"
    binary.write_text(FAKE_RUNNER)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    spaced = tmp_path / "a b"
    spaced.mkdir()
    eng = KrishivEngine(binary=binary)
    eng._tmp = spaced / "tmp dir"
    eng._tmp.mkdir()
    eng.setup(Dataset("tpch", 1, spaced, ["t"]))
    try:
        assert eng.timed_run(q("select 1"))[0] == 2
        assert eng.run(q("select 1")) == [(1, "x"), (2, "y")]
    finally:
        eng.teardown()


def test_duckdb_failure_in_timed_run_fails_only_that_query(engine: KrishivEngine):
    class Flaky(_Duck):
        calls = 0

        def run(self, query):
            Flaky.calls += 1
            if Flaky.calls > 1:
                raise RuntimeError("interrupted")
            return self.rows

    res = abba._one_query(q("select 1"), engine, Flaky([(1, "x"), (2, "y")]), 1, 5)
    assert res["status"] == "failed"
    assert "duckdb timed run RuntimeError" in res["reason"]
    assert "krishiv" not in res["reason"]


def test_zero_timeout_disables_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def fake_one(q, krishiv, duck, reps, timeout_s, duck_q=None):
        del duck_q
        seen["krishiv"] = krishiv.timeout_s
        seen["arg"] = timeout_s
        return {"status": "failed", "reason": "stub"}

    data = tmp_path / "data"
    ds = data / "tpch-duckdb" / "sf1"
    ds.mkdir(parents=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"a": [1]}), ds / "t.parquet")
    binary = tmp_path / "runner"
    binary.write_text(FAKE_RUNNER)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("KRISHIV_DATA", str(data))
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(binary))
    monkeypatch.setattr(abba, "load_queries", lambda *_: [Query("q01", "select 1", "tpch")])
    monkeypatch.setattr(abba, "_one_query", fake_one)
    abba.run_abba(1, timeout_s=0)
    assert seen == {"krishiv": None, "arg": 0}


def test_partitioned_dataset_registers_one_table(tmp_path):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krishiv_bench.abba import discover_tables
    from krishiv_bench.adapters.duckdb_engine import table_source

    parts = tmp_path / "hits"
    parts.mkdir()
    for i in range(3):
        pq.write_table(pa.table({"x": [i, i]}), parts / f"hits_{i}.parquet")
    pq.write_table(pa.table({"y": [1]}), tmp_path / "other.parquet")

    assert discover_tables(tmp_path) == ["hits", "other"]
    assert table_source(tmp_path, "hits").endswith("hits/*.parquet")
    assert table_source(tmp_path, "other").endswith("other.parquet")
    rel = duckdb.sql(f"select count(*) from read_parquet('{table_source(tmp_path, 'hits')}')")
    assert rel is not None
    assert rel.fetchall() == [(6,)]


def test_one_query_runs_duckdb_on_its_own_text(engine: KrishivEngine):
    seen: list[str] = []

    class Recording(_Duck):
        def run(self, query):
            seen.append(query.sql)
            return self.rows

    duck = Recording([(1, "x"), (2, "y")])
    done = abba._one_query(q("select 1"), engine, duck, 1, 5, q("select 1 -- duckdb text"))
    assert done["status"] == "ok"
    assert seen  # the correctness run and every timed run
    assert set(seen) == {"select 1 -- duckdb text"}


def test_run_abba_pairs_duckdb_texts_and_spark_null_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = tmp_path / "data"
    ds = data / "tpch-duckdb" / "sf1"
    ds.mkdir(parents=True)
    pq.write_table(pa.table({"a": [1]}), ds / "t.parquet")
    binary = tmp_path / "runner"
    binary.write_text(FAKE_RUNNER)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("KRISHIV_DATA", str(data))
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(binary))
    monkeypatch.setattr(abba, "wait_idle", lambda: None)

    def load(suite, engine, only):
        del suite, only
        extra = [Query("q02", "x", "tpch")] if engine == "krishiv" else []
        return [Query("q01", f"select 1 -- {engine}", "tpch"), *extra]

    seen: dict = {}

    def fake_one(q, krishiv, duck, reps, timeout_s, duck_q=None):
        del krishiv, reps, timeout_s
        seen[q.id] = (q.sql, duck_q.sql if duck_q else None, duck.null_order)
        return {"status": "failed", "reason": "stub"}

    monkeypatch.setattr(abba, "load_queries", load)
    monkeypatch.setattr(abba, "_one_query", fake_one)
    abba.run_abba(1, timeout_s=0)
    assert seen["q01"] == ("select 1 -- krishiv", "select 1 -- duckdb", "nulls_first_on_asc_last_on_desc")
    assert seen["q02"] == ("x", "x", "nulls_first_on_asc_last_on_desc")  # no DuckDB text: the shared one


def test_duckdb_null_order_matches_spark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krishiv_bench.adapters.duckdb_engine import SPARK_NULL_ORDER, DuckDBEngine

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    pq.write_table(pa.table({"x": [2, None, 1]}), tmp_path / "t.parquet")
    ds = Dataset("d", 1, tmp_path, ["t"])
    duck = DuckDBEngine(threads=1, memory_limit=None, null_order=SPARK_NULL_ORDER)
    duck.setup(ds)
    try:
        assert duck.run(Query("a", "select x from t order by x", "s")) == [(None,), (1,), (2,)]
        assert duck.run(Query("d", "select x from t order by x desc", "s")) == [(2,), (1,), (None,)]
    finally:
        duck.teardown()
    assert duck.config()["default_null_order"] == SPARK_NULL_ORDER
    assert DuckDBEngine(threads=1).config()["default_null_order"] == SPARK_NULL_ORDER  # the default
    assert "default_null_order" not in DuckDBEngine(threads=1, null_order=None).config()
    with pytest.raises(ValueError, match="null_order"):
        DuckDBEngine(null_order="sideways")
