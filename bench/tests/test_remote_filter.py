"""remote-filter suite: layout, thresholds, child runs on a tiny synthetic mirror, comparisons, memory cap."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from krishiv_bench import remote_filter
from krishiv_bench.memcap import Observation
from krishiv_bench.remote_filter import (
    compare_results,
    ledger_entry,
    load_expected,
    mirror_table_files,
    phase_buckets,
    run_one,
    run_remote_filter,
    write_expected,
)

ROWS = 2000


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int, int]:
    import numpy as np

    mirror = tmp_path_factory.mktemp("remote-filter") / "mirror"
    mirror.mkdir()
    rng = np.random.default_rng(7)
    table = pa.table(
        {
            "id": [f"<urn:uuid:{i:08d}>" for i in range(ROWS)],
            "text": [f"doc-{i}" for i in range(ROWS)],
            "int_score": rng.integers(0, 10, ROWS).astype("int64"),
            "token_count": rng.integers(0, 8000, ROWS).astype("int64"),
            "language_score": rng.uniform(0.8, 1.0, ROWS),
        }
    )
    pq.write_table(table, mirror / "fw.parquet", compression="zstd")
    keep = [
        i
        for i in range(ROWS)
        if table["int_score"][i].as_py() >= 4
        and table["token_count"][i].as_py() >= 4000
        and table["language_score"][i].as_py() >= 0.95
    ]
    expected_sum = sum(table["token_count"][i].as_py() for i in keep)
    assert len(keep) > 0
    return mirror, len(keep), expected_sum


def test_filter_thresholds():
    assert remote_filter.FILTER_SQL == "int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95"
    assert remote_filter.TABLE == "fw"
    assert remote_filter.ENGINES == ["krishiv", "duckdb", "polars"]
    assert "FineWeb-Edu" in remote_filter.describe()


def test_mirror_layout_and_expected_roundtrip(tiny, tmp_path):
    mirror, rows, total = tiny
    assert mirror_table_files(mirror) == [mirror / "fw.parquet"]
    assert load_expected(mirror) is None
    f = write_expected(mirror.parent / "x", rows, total, ["fw.parquet"])
    assert load_expected(mirror.parent / "x") == {"rows": rows, "sum_tokens": total, "files": ["fw.parquet"]}
    assert json.loads(f.read_text())["rows"] == rows
    with pytest.raises(FileNotFoundError, match="remote-filter --yes"):
        mirror_table_files(tmp_path / "empty")


def test_compare_results():
    ref = {"rows": 10, "sum_tokens": 100}
    assert compare_results(ref, {"rows": 10, "sum_tokens": 100}) is None
    assert "rows: reference 10, got 9" in (compare_results(ref, {"rows": 9, "sum_tokens": 100}) or "")
    assert "sum_tokens" in (compare_results(ref, {"rows": 10, "sum_tokens": 101}) or "")


def test_phase_buckets_carry_the_total_without_a_profile():
    assert phase_buckets({"query_s": 2.5}) == {"scan_s": None, "filter_s": None, "write_s": None, "total_s": 2.5}
    assert phase_buckets({})["total_s"] is None
    assert phase_buckets({"query_s": 1.0}, profile={"operators": "nope"})["scan_s"] is None


def test_phase_buckets_map_a_per_operator_profile():
    profile = {
        "operators": [
            {"name": "ParquetScan", "seconds": 1.0},
            {"name": "Filter", "seconds": 0.5},
            {"name": "ParquetWrite", "seconds": 0.25},
            {"name": 7, "seconds": "x"},
            "noise",
        ]
    }
    assert phase_buckets({"query_s": 2.0}, profile) == {
        "scan_s": 1.0,
        "filter_s": 0.5,
        "write_s": 0.25,
        "total_s": 2.0,
    }


def test_ledger_entry():
    rec = {
        "engine": "duckdb",
        "status": "finished",
        "pass": True,
        "peak_rss_bytes": 5,
        "phases": {"scan_s": None, "filter_s": None, "write_s": None, "total_s": 1.0},
    }
    assert ledger_entry(rec) == {
        "engine": "duckdb",
        "status": "finished",
        "pass": True,
        "scan_s": None,
        "filter_s": None,
        "write_s": None,
        "total_s": 1.0,
        "peak_rss_bytes": 5,
    }


def oracle_digest(mirror: Path) -> str:
    """sha256 of the sorted ids of the rows the filter keeps, computed row by row in Python."""
    import hashlib

    t = pq.read_table(mirror / "fw.parquet").to_pylist()
    ids = sorted(r["id"] for r in t if r["int_score"] >= 4 and r["token_count"] >= 4000 and r["language_score"] >= 0.95)
    h = hashlib.sha256()
    for v in ids:
        h.update(v.encode() + b"\n")
    return h.hexdigest()


def _result_of(capsys, out: Path) -> dict:
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("RESULT ")][-1]
    res = json.loads(line.removeprefix("RESULT "))
    assert (out / "result.parquet").is_file() or (out / "krishiv").is_dir()
    return res


def test_child_duckdb_matches_the_python_oracle(tiny, tmp_path, capsys):
    duckdb = pytest.importorskip("duckdb")
    del duckdb
    from krishiv_bench.remote_filter_child import main

    mirror, rows, total = tiny
    work = tmp_path / "w"
    assert main(["duckdb", str(mirror), str(work / "out"), str(work / "scratch")]) == 0
    res = _result_of(capsys, work / "out")
    assert (res["rows_out"], res["sum_tokens"]) == (rows, total)
    assert res["ids_digest"] == oracle_digest(mirror)
    assert res["peak_rss_bytes"] > 0
    assert "ZSTD" in res["compression"]
    assert res["query_s"] >= 0


def test_child_polars_matches_the_python_oracle(tiny, tmp_path, capsys):
    pytest.importorskip("polars")
    from krishiv_bench.remote_filter_child import main

    mirror, rows, total = tiny
    work = tmp_path / "w"
    assert main(["polars", str(mirror), str(work / "out"), str(work / "scratch")]) == 0
    res = _result_of(capsys, work / "out")
    assert (res["rows_out"], res["sum_tokens"]) == (rows, total)
    assert res["ids_digest"] == oracle_digest(mirror)
    assert "ZSTD" in res["compression"]


def test_child_rejects_bad_input(tiny, tmp_path):
    from krishiv_bench.remote_filter_child import main

    mirror, _, _ = tiny
    with pytest.raises(SystemExit, match="unknown remote-filter engine 'flink'"):
        main(["flink", str(mirror), str(tmp_path / "o"), str(tmp_path / "s")])
    with pytest.raises(SystemExit, match="local mirror"):
        main(["krishiv", '["https://example.com/a.parquet"]', str(tmp_path / "o"), str(tmp_path / "s")])
    with pytest.raises(SystemExit, match="non-empty list"):
        main(["duckdb", "[]", str(tmp_path / "o"), str(tmp_path / "s")])
    with pytest.raises(SystemExit, match="no Parquet files"):
        main(["duckdb", str(tmp_path), str(tmp_path / "o"), str(tmp_path / "s")])


def _runner_binary() -> Path | None:
    import os

    env = os.environ.get("KRISHIV_BENCH_RUNNER")
    if env:
        p = Path(env)
        return p if p.exists() else None
    p = Path(__file__).resolve().parents[2] / "target" / "dist" / "krishiv-bench-runner"
    return p if p.exists() else None


def test_child_krishiv_matches_the_oracle_when_the_runner_exists(tiny, tmp_path, capsys):
    if _runner_binary() is None:
        pytest.skip("krishiv-bench-runner not built")
    from krishiv_bench.remote_filter_child import main

    mirror, rows, total = tiny
    work = tmp_path / "w"
    assert main(["krishiv", str(mirror), str(work / "out"), str(work / "scratch")]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("RESULT ")][-1]
    res = json.loads(line.removeprefix("RESULT "))
    assert (res["rows_out"], res["sum_tokens"]) == (rows, total)


def test_child_krishiv_passes_only_supported_conf_to_the_runner(tmp_path, monkeypatch, capsys):
    import stat
    import sys
    import textwrap

    argv_log = tmp_path / "argv.json"
    runner = tmp_path / "runner"
    runner.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, os, sys
            from urllib.parse import unquote
            import pyarrow as pa, pyarrow.parquet as pq
            json.dump(sys.argv[1:], open({str(argv_log)!r}, "w"))
            print("READY 9.9.9 fw", flush=True)
            for line in sys.stdin:
                cmd = line.split()
                if cmd[0] == "QUIT":
                    break
                out = unquote(cmd[2]); os.makedirs(out)
                pq.write_table(pa.table({{"text": ["a", "b"], "int_score": [4, 9],
                    "token_count": [4000, 5000], "language_score": [0.96, 0.99]}}), out + "/part-0.parquet")
                print("OK 2 0.5", flush=True)
            """
        )
    )
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(runner))
    from krishiv_bench.remote_filter_child import main

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    pq.write_table(
        pa.table({"text": ["a"], "int_score": [9], "token_count": [5000], "language_score": [0.99]}),
        mirror / "fw.parquet",
    )
    work = tmp_path / "w"
    assert main(["krishiv", str(mirror), str(work / "out"), str(work / "scratch")]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("RESULT ")][-1]
    res = json.loads(line.removeprefix("RESULT "))
    assert (res["rows_out"], res["sum_tokens"]) == (2, 9000)
    assert res["compression"] == ["SNAPPY"]  # the fake runner writes PyArrow's default codec
    args = json.loads(argv_log.read_text())
    assert not any(a.startswith("spark.sql.") for a in args)
    assert f"krishiv.spill.dir={work / 'scratch'}" in args


# ------------------------------------------------------------------ run_one with scripted processes
class FakeProc:
    def __init__(self, pid: int, returncode: int | None, timed_out: bool = False):
        self.pid, self.returncode, self._timed_out = pid, returncode, timed_out

    def poll(self):
        return self.returncode


def fake_popen(result_line: str | None, returncode: int | None):
    def go(cmd, *, stdout, stderr, env, stdin, start_new_session=False):
        if result_line is not None:
            stdout.write((result_line + "\n").encode())
        return FakeProc(1234, returncode)

    return go


def obs_of(returncode: int | None, timed_out=False, peak=0, oom=0, rss=700):
    return Observation(returncode, timed_out, peak, oom, None, 1.5, rss)


def test_run_one_finished_records_peak_rss_and_phases(tmp_path):
    line = 'RESULT {"version": "1", "setup_s": 0.1, "query_s": 2.0, "rows_out": 10, "sum_tokens": 100}'
    rec = run_one(
        "duckdb",
        '["https://example.com/a.parquet"]',
        None,
        60,
        tmp_path / "w",
        _popen=fake_popen(line, 0),
        _find_cgroup=lambda pid: None,
        _monitor=lambda p, c, s, t: obs_of(0),
    )
    assert rec["status"] == "finished"
    assert (rec["rows_out"], rec["sum_tokens"]) == (10, 100)
    assert rec["peak_rss_bytes"] == 700
    assert rec["phases"] == {"scan_s": None, "filter_s": None, "write_s": None, "total_s": 2.0}


def test_run_one_simulated_oom_is_killed_not_a_crash(tmp_path):
    from krishiv_bench.sizes import parse_size

    cap_bytes = parse_size("64m")
    rec = run_one(
        "polars",
        '["https://example.com/a.parquet"]',
        "64m",
        60,
        tmp_path / "w",
        _popen=fake_popen(None, -9),
        _find_cgroup=lambda pid: None,
        _monitor=lambda p, c, s, t: obs_of(-9, peak=cap_bytes, oom=1),
    )
    assert rec["status"] == "killed"
    assert rec["oom_kills"] == 1
    assert "pass" not in rec  # the suite assigns pass/fail; the kill itself is the recorded result


def test_run_one_timeout_and_failure(tmp_path):
    rec = run_one(
        "duckdb",
        '["https://example.com/a.parquet"]',
        None,
        1,
        tmp_path / "wt",
        _popen=fake_popen(None, -9),
        _find_cgroup=lambda pid: None,
        _monitor=lambda p, c, s, t: obs_of(-9, timed_out=True),
    )
    assert rec["status"] == "timeout"
    rec = run_one(
        "duckdb",
        '["https://example.com/a.parquet"]',
        None,
        60,
        tmp_path / "wf",
        _popen=fake_popen(None, 1),
        _find_cgroup=lambda pid: None,
        _monitor=lambda p, c, s, t: obs_of(1),
    )
    assert rec["status"] == "failed"
    assert rec["returncode"] == 1


def test_run_one_rejects_bad_engine_and_cap(tmp_path):
    with pytest.raises(ValueError, match="non-empty subset"):
        run_one("flink", "[]", None, 60, tmp_path / "w")
    with pytest.raises(ValueError, match="memory size must"):
        run_one("duckdb", "[]", "lots", 60, tmp_path / "w")


# ------------------------------------------------------------------ suite orchestration
def _script(results: dict[str, dict]):
    def go(engine, data_arg, cap, timeout_s, work: Path) -> dict:
        base = {"engine": engine, "wall_s": 1.0, "peak_rss_bytes": 8, "peak_cgroup_bytes": 9}
        r = dict(results[engine])
        if r["status"] == "finished":
            r["phases"] = phase_buckets(r)
        return base | r

    return go


def _fin(rows=5, tokens=50, ids="d1", wall=1.0):
    return {"status": "finished", "query_s": wall, "rows_out": rows, "sum_tokens": tokens, "ids_digest": ids}


def _scripted(results: dict[str, dict], calls: list | None = None):
    def go(engine, data_arg, cap, timeout_s, work: Path) -> dict:
        if calls is not None:
            calls.append((cap, engine))
        base = {"engine": engine, "wall_s": 1.0, "peak_rss_bytes": 8, "peak_cgroup_bytes": 9}
        return base | dict(results[engine])

    return go


def test_suite_runs_validation_then_reps_with_rotation_and_duckdb_as_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    calls: list = []
    f = run_remote_filter(
        "[]",
        ["krishiv", "duckdb", "polars"],
        [None, "4g"],
        60,
        run=_scripted({e: _fin() for e in ("krishiv", "duckdb", "polars")}, calls),
        reps=2,
    )
    record = json.loads(f.read_text())
    assert record["reps"] == 2
    assert set(record["scopes"]) == {"uncapped", "4g"}
    # 3 runs per rep, (validation + 2 timed) reps, 2 scopes; DuckDB first in each validation pass.
    assert len(calls) == 3 * 3 * 2
    assert calls[0] == (None, "duckdb")
    assert calls[9] == ("4g", "duckdb")
    # The engine order rotates between timed repetitions: not the same engine first twice in a row.
    first_scope = [e for _, e in calls[:9]]
    assert first_scope == ["duckdb", "krishiv", "polars", "duckdb", "polars", "krishiv", "polars", "krishiv", "duckdb"]
    unc = record["scopes"]["uncapped"]
    assert unc["reference"] == {"rows": 5, "sum_tokens": 50, "ids_digest": "d1"}
    assert unc["engines"]["duckdb"]["verdict"] == "reference"
    assert unc["engines"]["krishiv"]["verdict"] == "pass"
    assert unc["engines"]["krishiv"]["summary"]["n"] == 2
    assert unc["engines"]["krishiv"]["summary"]["median_wall_s"] == 1.0
    assert unc["engines"]["krishiv"]["wall_vs_duckdb"]["base"] == "duckdb"
    assert unc["engines"]["duckdb"]["wall_vs_duckdb"] is None
    assert record["summary"] == {"n": 6, "finished": 12, "killed": 0, "passed": 12}


def test_suite_flags_a_wrong_result_and_never_times_it(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    calls: list = []
    f = run_remote_filter(
        "[]",
        ["krishiv", "duckdb", "polars"],
        ["64m"],
        60,
        run=_scripted(
            {
                "duckdb": _fin(),
                "polars": _fin(tokens=51),
                "krishiv": {"status": "killed", "returncode": -9, "oom_kills": 1},
            },
            calls,
        ),
        reps=2,
    )
    record = json.loads(f.read_text())
    sc = record["scopes"]["64m"]["engines"]
    assert sc["polars"]["verdict"].startswith("fail: sum_tokens")
    assert [r["status"] for r in sc["polars"]["runs"]] == ["mismatch"]  # validation only, no timed runs
    assert sc["polars"]["summary"]["n"] == 0
    assert sc["krishiv"]["verdict"] == "unverified: killed"
    assert sc["krishiv"]["summary"]["killed"] == 2
    assert record["summary"] == {"n": 3, "finished": 2, "killed": 2, "passed": 2}
    assert f.name.endswith("_remote-filter.json")


def test_an_engine_that_returns_other_ids_with_the_same_counts_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    f = run_remote_filter(
        "[]",
        ["polars", "duckdb"],
        None,
        60,
        run=_scripted({"duckdb": _fin(ids="a"), "polars": _fin(ids="b")}),
        reps=1,
    )
    polars = json.loads(f.read_text())["scopes"]["uncapped"]["engines"]["polars"]
    assert polars["verdict"].startswith("fail: ids_digest")


def test_expected_json_is_the_reference_and_no_reference_is_unverified(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    pq.write_table(pa.table({"text": ["a"]}), mirror / "fw.parquet")
    write_expected(mirror, 5, 50, [])
    f = run_remote_filter(str(mirror), ["polars"], None, 60, run=_scripted({"polars": _fin(tokens=49)}), reps=1)
    sc = json.loads(f.read_text())["scopes"]["uncapped"]
    assert sc["reference"] == {"rows": 5, "sum_tokens": 50, "files": []}
    assert sc["engines"]["polars"]["verdict"].startswith("fail: sum_tokens")
    # DuckDB passing against the counts completes the reference with its id digest, which then binds the others.
    f = run_remote_filter(
        str(mirror),
        ["duckdb", "polars"],
        None,
        60,
        run=_scripted({"duckdb": _fin(), "polars": _fin(ids="other")}),
        reps=1,
    )
    sc = json.loads(f.read_text())["scopes"]["uncapped"]
    assert sc["reference"]["ids_digest"] == "d1"
    assert sc["engines"]["polars"]["verdict"].startswith("fail: ids_digest")
    # A mirror that was never downloaded fails before any scope starts.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="krishiv-datasets remote-filter"):
        run_remote_filter(str(empty), ["duckdb"], None, 60, run=_scripted({"duckdb": _fin()}), reps=1)
    # Without DuckDB and without counts nothing can be checked.
    f = run_remote_filter("[]", ["polars"], None, 60, run=_scripted({"polars": _fin()}), reps=1)
    polars = json.loads(f.read_text())["scopes"]["uncapped"]["engines"]["polars"]
    assert polars["verdict"] == "unverified: no reference result"
    assert polars["runs"][0]["pass"] is None


def test_the_result_is_written_when_a_run_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    ok = _scripted({"duckdb": _fin()})

    def boom(engine, data_arg, cap, timeout_s, work):
        if cap == "4g":
            raise OSError("no systemd-run")
        return ok(engine, data_arg, cap, timeout_s, work)

    with pytest.raises(OSError, match="no systemd-run"):
        run_remote_filter("[]", ["duckdb"], [None, "4g"], 60, run=boom, reps=1)
    (path,) = (tmp_path / "results").glob("*_remote-filter.json")
    record = json.loads(path.read_text())
    assert record["error"].startswith("OSError")
    assert "uncapped" in record["scopes"]  # the finished scope is kept
    assert "4g" not in record["scopes"]


def test_suite_validates_engines_and_reps(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="non-empty subset"):
        run_remote_filter("[]", ["flink"], None, 60, run=_scripted({}))
    with pytest.raises(ValueError, match="reps must be at least 1"):
        run_remote_filter("[]", ["duckdb"], None, 60, run=_scripted({}), reps=0)


def test_compare_results_ignores_unknown_reference_keys_and_checks_ids():
    assert compare_results({"rows": 1, "sum_tokens": 2}, {"rows": 1, "sum_tokens": 2, "ids_digest": "x"}) is None
    diff = compare_results(
        {"rows": 1, "sum_tokens": 2, "ids_digest": "a"}, {"rows": 1, "sum_tokens": 2, "ids_digest": "b"}
    )
    assert diff is not None
    assert "ids_digest" in diff


def test_listing_pins_the_commit_and_fetch_resumes_through_a_part_file(monkeypatch, tmp_path):
    import io

    from krishiv_bench import remote_filter as rf

    listing = {
        "sha": "abc123",
        "siblings": [
            {"rfilename": "sample/10BT/000_00000.parquet"},
            {"rfilename": "sample/10BT/README.md"},
            {"rfilename": "other/x.parquet"},
        ],
    }
    monkeypatch.setattr(rf.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(json.dumps(listing).encode()))
    ((url, rel),) = rf.list_remote_files()
    assert "/resolve/abc123/" in url
    assert rel == "sample/10BT/000_00000.parquet"
    monkeypatch.setattr(rf.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b'{"siblings": []}'))
    with pytest.raises(RuntimeError, match="no commit sha"):
        rf.list_remote_files()
    monkeypatch.setattr(rf.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b'{"sha": "abc", "siblings": []}'))
    with pytest.raises(RuntimeError, match="no Parquet files"):
        rf.list_remote_files()
    monkeypatch.setattr(rf.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b"[1]"))
    with pytest.raises(RuntimeError, match="unexpected Hub listing"):
        rf.list_remote_files()

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"x")  # curl writes the .part file

    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    dest = rf.fetch_remote_filter(tmp_path / "m", yes=True, files=[("https://h/a.parquet", "a.parquet")])
    assert calls[0][calls[0].index("-o") + 1].endswith("a.parquet.part")
    assert calls[0][calls[0].index("--speed-limit") + 1] == "1000"  # a stalled transfer fails and retries
    assert "--connect-timeout" in calls[0]
    assert (dest / "fw" / "a.parquet").is_file()
    assert not (dest / "fw" / "a.parquet.part").exists()
    # A finished file is kept; no second download.
    rf.fetch_remote_filter(tmp_path / "m", yes=True, files=[("https://h/a.parquet", "a.parquet")])
    assert len(calls) == 1


def test_uncapped_peak_rss_comes_from_the_child(tmp_path):
    line = 'RESULT {"engine": "duckdb", "rows_out": 1, "sum_tokens": 2, "peak_rss_bytes": 4096, "query_s": 1.0}'
    rec = run_one(
        "duckdb",
        "[]",
        None,
        60,
        tmp_path / "w",
        _popen=fake_popen(line, 0),
        _monitor=lambda proc, cg, _x, _t: obs_of(0, rss=0),
    )
    assert rec["peak_rss_bytes"] == 4096


def test_a_timeout_kills_a_grandchild_in_the_real_process_group(monkeypatch, tmp_path):
    import time

    from krishiv_bench import remote_filter as rf

    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "sleeper_child.py").write_text(
        "import subprocess, sys, time\n"
        "g = subprocess.Popen(['sleep', '300'])\n"
        "open(sys.argv[2], 'w').write(str(g.pid))\n"
        "time.sleep(300)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(mods))
    monkeypatch.setattr(rf, "CHILD", "sleeper_child")
    pid_file = tmp_path / "grandchild.pid"
    rec = rf.run_one("duckdb", str(pid_file), None, 1, tmp_path / "w")
    assert rec["status"] == "timeout"
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_krishiv_memory_limit_is_75_percent_of_the_cap(tmp_path):
    import inspect

    from krishiv_bench.adapters.krishiv import KrishivEngine
    from krishiv_bench.remote_filter import krishiv_memory_limit

    default = inspect.signature(KrishivEngine.__init__).parameters["memory_limit"].default
    assert krishiv_memory_limit("4g") == "3072m"
    assert krishiv_memory_limit(None) == default

    seen: dict = {}

    def go_capped(cmd, *, stdout, stderr, env, stdin, start_new_session=False):
        seen["capped"] = cmd
        stdout.write(
            b'RESULT {"version": "1", "setup_s": 0.1, "query_s": 2.0, "rows_out": 10, '
            b'"sum_tokens": 100, "engine_memory_limit": "3072m"}\n'
        )
        return FakeProc(0, 0)

    rec = run_one(
        "krishiv",
        '["https://example.com/a.parquet"]',
        "4g",
        60,
        tmp_path / "w-cap",
        _popen=go_capped,
        _find_cgroup=lambda pid: None,
        _monitor=lambda p, c, s, t: obs_of(0),
    )
    assert seen["capped"][-1] == "3072m"
    assert rec["engine_memory_limit"] == "3072m"

    def go_plain(cmd, *, stdout, stderr, env, stdin, start_new_session=False):
        seen["plain"] = cmd
        line = (
            f'RESULT {{"version": "1", "setup_s": 0.1, "query_s": 2.0, "rows_out": 10, '
            f'"sum_tokens": 100, "engine_memory_limit": "{default}"}}\n'
        )
        stdout.write(line.encode())
        return FakeProc(0, 0)

    rec = run_one(
        "krishiv",
        '["https://example.com/a.parquet"]',
        None,
        60,
        tmp_path / "w-plain",
        _popen=go_plain,
        _find_cgroup=lambda pid: None,
        _monitor=lambda p, c, s, t: obs_of(0),
    )
    assert len(seen["plain"]) == 7
    assert rec["engine_memory_limit"] == default


# ------------------------------------------------------------------ download
def test_fetch_needs_opt_in(tmp_path):
    with pytest.raises(SystemExit, match="~28 GB"):
        remote_filter.fetch_remote_filter(tmp_path / "x")


def test_fetch_downloads_explicit_files(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    seen = []

    def fake_run(cmd, check):
        assert cmd[0] == "curl"
        seen.append(cmd[-1])
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"PAR1fake")

    monkeypatch.setattr(subprocess, "run", fake_run)
    dest = remote_filter.fetch_remote_filter(
        tmp_path / "mirror", yes=True, files=[("https://example.com/p0.parquet", "p0.parquet")]
    )
    assert (dest / "fw" / "p0.parquet").is_file()
    assert (dest / remote_filter.EXPECTED_FILE).is_file()
    assert seen == ["https://example.com/p0.parquet"]
    with pytest.raises(ValueError, match="no files"):
        remote_filter.fetch_remote_filter(tmp_path / "mirror2", yes=True, files=[])


def test_parse_input_accepts_urls_files_directories_and_refuses_the_rest(tmp_path):
    from krishiv_bench.remote_filter_child import parse_input

    assert parse_input('["https://h/a.parquet"]') == ("urls", ["https://h/a.parquet"])
    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(pa.table({"text": ["a"]}), d / "x.parquet")
    assert parse_input(json.dumps([str(d), "/elsewhere/y.parquet"])) == (
        "files",
        [str(d / "x.parquet"), "/elsewhere/y.parquet"],
    )
    assert parse_input(str(d)) == ("files", [str(d / "x.parquet")])
    assert parse_input(str(d / "x.parquet")) == ("files", [str(d / "x.parquet")])
    empty = tmp_path / "empty"
    empty.mkdir()
    for bad, want in [
        ("[1]", "non-empty list of strings"),
        ("[]", "non-empty list of strings"),
        (str(empty), "no Parquet files"),
        (str(tmp_path / "missing"), "neither a directory"),
        (json.dumps([str(empty)]), "no Parquet sources"),
    ]:
        with pytest.raises(SystemExit, match=want):
            parse_input(bad)


def test_the_cli_commands_forward_and_report_errors(monkeypatch, tmp_path):
    from krishiv_bench import cli, datasets
    from krishiv_bench import remote_filter as rf

    seen: dict = {}

    def fake_run(data_arg, engines, caps, timeout_s, reps):
        seen.update(data=data_arg, engines=engines, caps=caps, timeout=timeout_s, reps=reps)
        return tmp_path / "r.json"

    monkeypatch.setattr(rf, "run_remote_filter", fake_run)
    cli.main(["remote-filter", "--mirror", str(tmp_path), "--engines", "duckdb, polars", "--reps", "2"])
    assert seen["data"] == str(tmp_path)
    assert seen["engines"] == ["duckdb", "polars"]
    assert seen["caps"] == [None, "4g"]
    assert seen["reps"] == 2

    def refuse(*_a, **_k):
        raise FileNotFoundError("no mirror")

    monkeypatch.setattr(rf, "run_remote_filter", refuse)
    with pytest.raises(SystemExit) as e:
        cli.main(["remote-filter"])
    assert e.value.code == 2

    monkeypatch.setattr(rf, "fetch_remote_filter", lambda out, yes: tmp_path / "m")
    datasets.main(["remote-filter", "--yes"])

    def broken(out, yes):
        raise RuntimeError("cannot list")

    monkeypatch.setattr(rf, "fetch_remote_filter", broken)
    with pytest.raises(SystemExit, match="cannot list"):
        datasets.main(["remote-filter", "--yes"])


def test_a_new_revision_clears_the_old_mirror_and_the_same_one_keeps_it(monkeypatch, tmp_path):
    from krishiv_bench import remote_filter as rf

    def listing(sha):
        return [(f"https://h/datasets/x/resolve/{sha}/a.parquet", "a.parquet")]

    def fake_run(cmd, check):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"x")

    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    monkeypatch.setattr(rf, "list_remote_files", lambda: listing("r1"))
    dest = rf.fetch_remote_filter(tmp_path / "m", yes=True)
    assert (dest / "fw" / ".revision").read_text() == "r1\n"
    (dest / "fw" / "a.parquet").write_bytes(b"old")
    (dest / "fw" / "b.parquet.part").write_bytes(b"partial")
    rf.fetch_remote_filter(tmp_path / "m", yes=True)  # same revision: the finished file is kept
    assert (dest / "fw" / "a.parquet").read_bytes() == b"old"
    assert (dest / "fw" / "b.parquet.part").exists()
    monkeypatch.setattr(rf, "list_remote_files", lambda: listing("r2"))
    rf.fetch_remote_filter(tmp_path / "m", yes=True)  # a new revision: old files go, the new one is fetched
    assert (dest / "fw" / "a.parquet").read_bytes() == b"x"
    assert not (dest / "fw" / "b.parquet.part").exists()
    assert (dest / "fw" / ".revision").read_text() == "r2\n"
    # Several revisions in one listing cannot be pinned: nothing is cleared.
    mixed = [*listing("r2"), *listing("r3")]
    rf._pin_revision(dest, dest / "fw", mixed)
    assert (dest / "fw" / ".revision").read_text() == "r2\n"
