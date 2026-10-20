"""Peak memory per engine and query, Krishiv's profile mode, and attribution of timings to queries."""

from __future__ import annotations

import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from krishiv_bench import reproduce
from krishiv_bench.adapters import Dataset, Engine, Query
from krishiv_bench.adapters.krishiv import KrishivEngine
from krishiv_bench.adapters.proc_mem import TreeProc, descendants
from krishiv_bench.adapters.process import ProcessEngine
from krishiv_bench.runner import load_queries

MIB = 2**20


def _spawn_holder(mib: int, grandchild_mib: int = 0) -> subprocess.Popen[bytes]:
    """A python process holding ``mib`` MiB of touched memory (and a child holding ``grandchild_mib``)."""
    code = textwrap.dedent(
        f"""
        import subprocess, sys, time
        if {grandchild_mib}:
            subprocess.Popen([sys.executable, "-c",
                "import time; b = bytearray({grandchild_mib} * 2**20); b[::4096] = b'x' * len(b[::4096]); "
                "time.sleep(60)"])
        b = bytearray({mib} * 2**20)
        b[::4096] = b"x" * len(b[::4096])
        print("ready", flush=True)
        time.sleep(60)
        """
    )
    p = subprocess.Popen([sys.executable, "-I", "-c", code], stdout=subprocess.PIPE)
    assert p.stdout is not None
    assert p.stdout.readline().strip() == b"ready"
    time.sleep(0.5)  # let a grandchild finish touching its memory
    return p


def test_tree_peak_counts_descendants():
    p = _spawn_holder(80, grandchild_mib=120)
    try:
        assert len(descendants(p.pid)) >= 2
        tree = TreeProc(p.pid)
        tree.start()
        peak = tree.stop()
        # parent 80 MiB + grandchild 120 MiB of touched pages (plus interpreter baselines)
        assert peak >= 190 * MIB, peak / MIB
        solo = _spawn_holder(80)
        try:
            only = TreeProc(solo.pid)
            only.start()
            assert only.stop() < 190 * MIB
        finally:
            solo.kill()
            solo.wait()
    finally:
        p.kill()
        p.wait()


def test_tree_peak_sees_a_spike_inside_the_window():
    code = textwrap.dedent(
        """
        import sys, time
        sys.stdin.readline()
        b = bytearray(150 * 2**20); b[::4096] = b"x" * len(b[::4096])
        del b
        print("done", flush=True)
        time.sleep(60)
        """
    )
    p = subprocess.Popen([sys.executable, "-I", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert p.stdin is not None
        assert p.stdout is not None
        tree = TreeProc(p.pid, interval_s=0.5)  # sparse sampling: the VmHWM reset/read still catches the spike
        time.sleep(0.5)
        tree.start()
        p.stdin.write(b"go\n")
        p.stdin.flush()
        assert p.stdout.readline().strip() == b"done"
        assert tree.stop() >= 150 * MIB
    finally:
        p.kill()
        p.wait()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Dataset:
    import duckdb

    root = tmp_path_factory.mktemp("tinymem")
    duckdb.sql(f"COPY (SELECT range AS a, range % 3 AS g FROM range(1000)) TO '{root / 't.parquet'}' (FORMAT parquet)")
    return Dataset("tiny", 1, root, ["t"])


def test_process_engine_records_peak_rss_of_its_worker(tiny):
    eng = ProcessEngine("duckdb", 2, 1, 60)
    eng.setup(tiny)
    try:
        assert eng.last_peak_rss_bytes is None
        eng.timed_run(Query("q01", "SELECT g, count(*) FROM t GROUP BY g", "tpch"))
        assert eng.last_peak_rss_bytes is not None
        assert eng.last_peak_rss_bytes > 20 * MIB  # a python + duckdb worker is never smaller
    finally:
        eng.teardown()
    assert eng.end_measure().peak_rss_bytes is None  # no window open


def test_reproduce_records_peak_per_engine_and_query_in_json_and_table(tiny):
    engines = {n: ProcessEngine(n, 2, 1, 60) for n in ("duckdb", "polars")}
    for e in engines.values():
        e.setup(tiny)
    try:
        qm = {n: Query("q01", "SELECT g, count(*) AS c FROM t GROUP BY g", "tpch") for n in engines}
        ent = reproduce.ours_query(qm, engines, reps=1, timeout_s=60)
    finally:
        for e in engines.values():
            e.teardown()
    for n in engines:
        assert ent["engines"][n]["peak_rss_bytes"] > 20 * MIB
    theirs = reproduce.theirs_query("duckdb", qm["duckdb"], tiny, 2, 1, 2, 60, 3)
    assert theirs["peak_rss_bytes"] > 20 * MIB
    ent["engines"]["duckdb"]["theirs"] = theirs
    record = {
        "methods": ["ours", "theirs"],
        "suite": "tpch",
        "dataset": {"scale_factor": 1},
        "pinned": {"threads": 2},
        "engines": {"duckdb": {}, "polars": {}},
        "queries": {"q01": ent},
        "load_at_start": 0.5,
    }
    record["summary"] = reproduce.summarize(record)
    assert record["summary"]["ours"]["polars"]["max_peak_rss_bytes"] == ent["engines"]["polars"]["peak_rss_bytes"]
    assert record["summary"]["theirs"]["duckdb"]["max_peak_rss_bytes"] == theirs["peak_rss_bytes"]
    md = reproduce.markdown(record)
    assert "**ours peak memory** (MiB" in md
    assert "**theirs peak memory** (MiB" in md
    lines = md.splitlines()
    row = lines[lines.index("**ours peak memory** (MiB, RSS of each engine's whole process tree)") + 4]
    assert row.startswith("| q01 |")
    assert float(row.split("|")[2]) == pytest.approx(ent["engines"]["duckdb"]["peak_rss_bytes"] / MIB, abs=1)


def test_peak_table_marks_engines_without_a_measurement():
    record = {
        "queries": {"q01": {"engines": {"a": {"peak_rss_bytes": 3 * MIB}, "b": {"status": "error"}}}},
    }
    assert reproduce._peak_table(record, "ours", ["a", "b"])[-2] == "| q01 | 3 | - |"
    assert reproduce._peak_table({"queries": {"q01": {}}}, "ours", ["a"]) == []


class _Profiled(Engine):
    name = "krishiv"

    def __init__(self, fail: bool = False):
        self.fail = fail

    def version(self) -> str:
        return "p"

    def setup(self, dataset: Dataset) -> None:
        pass

    def run(self, query: Query):
        return [(1,)]

    def profile(self, query: Query) -> dict:
        if self.fail:
            raise NotImplementedError("no profile")
        return {"rows": 1, "seconds": 0.5, "phases": [{"name": "plan_execute", "seconds": 0.4}]}


def test_ours_query_attaches_profile_only_when_asked():
    class Ref(_Profiled):
        name = "duckdb"

    engines = {"duckdb": Ref(), "krishiv": _Profiled()}
    qm = {n: Query("q01", "x", "tpch") for n in engines}
    off = reproduce.ours_query(qm, engines, reps=1, timeout_s=5)
    assert "profile" not in off["engines"]["krishiv"]
    on = reproduce.ours_query(qm, engines, reps=1, timeout_s=5, profile=True)
    assert on["engines"]["krishiv"]["profile"]["phases"][0]["name"] == "plan_execute"
    engines["krishiv"] = _Profiled(fail=True)
    bad = reproduce.ours_query(qm, engines, reps=1, timeout_s=5, profile=True)
    assert bad["engines"]["krishiv"]["profile_error"] == "no profile"
    assert bad["engines"]["krishiv"]["status"] == "ok"


# A stand-in runner speaking the line protocol: time and rows are derived from the SQL text it reads from the
# file named in the command, so a mix-up of query ids, files or replies shows up in the numbers.
_FAKE_RUNNER = """#!PYTHON
import json, sys, urllib.parse
print("READY 0 t", flush=True)
for line in sys.stdin:
    cmd, *args = line.split()
    if cmd == "QUIT":
        break
    sql = open(urllib.parse.unquote(args[0])).read()
    secs = int(sql.split()[1]) / 1000.0
    rows = len(sql)
    if cmd == "PROFILE":
        detail = json.dumps({"phases": [{"name": "x", "seconds": secs}]})
        print(f"OK {rows} {secs:.6f} {detail}", flush=True)
    else:
        print(f"OK {rows} {secs:.6f}", flush=True)
"""


@pytest.fixture
def fake_runner(tmp_path: Path) -> Path:
    f = tmp_path / "runner"
    f.write_text(_FAKE_RUNNER.replace("PYTHON", sys.executable))
    f.chmod(f.stat().st_mode | stat.S_IXUSR)
    return f


def test_krishiv_timings_follow_the_query_that_ran(fake_runner: Path, tmp_path: Path):
    eng = KrishivEngine(threads=1, binary=fake_runner)
    eng.setup(Dataset("tpch", 10, tmp_path, ["t"]))
    try:
        q18 = Query("q18", "SELECT 5900 q18", "tpch")
        q21 = Query("q21", "SELECT 1200 -- q21 with a longer text", "tpch")
        for _ in range(2):  # alternating, repeated: no reply or file is reused for the wrong query
            assert eng.timed_run(q18) == (len(q18.sql), 5.9)
            assert eng.timed_run(q21) == (len(q21.sql), 1.2)
        p = eng.profile(q21)
        assert (p["rows"], p["seconds"], p["phases"][0]["seconds"]) == (len(q21.sql), 1.2, 1.2)
    finally:
        eng.teardown()


def test_same_query_id_with_new_text_runs_the_new_text(fake_runner: Path, tmp_path: Path):
    """The SQL file is named after the query id; a second query with the same id must not run the first's text."""
    eng = KrishivEngine(threads=1, binary=fake_runner)
    eng.setup(Dataset("tpch", 10, tmp_path, ["t"]))
    try:
        assert eng.timed_run(Query("q01", "SELECT 100", "tpch"))[1] == 0.1
        assert eng.timed_run(Query("q01", "SELECT 200", "tpch"))[1] == 0.2
    finally:
        eng.teardown()


def test_q18_and_q21_are_distinct_queries_for_every_engine():
    for engine in ("krishiv", "duckdb", "polars", "datafusion", "sail", "spark-local"):
        qs = {q.id: q.sql for q in load_queries("tpch", engine, ["q18", "q21"])}
        assert set(qs) == {"q18", "q21"}
        assert qs["q18"] != qs["q21"]
        assert "c_name" in qs["q18"]  # large volume customer
        assert "s_name" in qs["q21"]  # suppliers who kept orders waiting


def test_the_timed_span_is_one_request(tiny):
    """A worker's reported time covers only its own request, so queries timed back to back are not cumulative."""
    eng = ProcessEngine("duckdb", 2, 1, 60)
    eng.setup(tiny)
    try:
        slow = Query("q18", "SELECT count(*) FROM t a, t b, t c WHERE a.a % 7 = b.a % 7 AND b.a % 5 = c.a % 5", "tpch")
        fast = Query("q21", "SELECT 1", "tpch")
        _, t_slow = eng.timed_run(slow)
        _, t_fast = eng.timed_run(fast)
        _, t_fast2 = eng.timed_run(fast)
        assert t_slow > 20 * t_fast
        assert t_fast2 < t_slow / 20
    finally:
        eng.teardown()
