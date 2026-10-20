"""Build-vs-build A/B: CLI parsing, correctness gate, ABBA order and --conf passthrough (fake runner)."""

from __future__ import annotations

import json
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from krishiv_bench import ab, cli
from krishiv_bench.adapters.base import Dataset, Query
from krishiv_bench.adapters.krishiv import KrishivEngine, runner_binary

# A runner that records its argv, supports --build-info only when asked to, and returns a result that embeds the
# value of the "mode" conf, so two runs with different confs produce different rows.
FAKE = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import json, os, sys
    from urllib.parse import unquote
    import pyarrow as pa, pyarrow.parquet as pq
    argv = sys.argv[1:]
    if argv == ["--build-info"]:
        if os.path.exists(__file__ + ".buildinfo"):
            print(open(__file__ + ".buildinfo").read())
            sys.exit(0)
        sys.stderr.write("unknown argument --build-info\\n")
        sys.exit(1)
    open(__file__ + ".argv", "w").write(json.dumps(argv))
    mode = "default"
    for i, x in enumerate(argv):
        if x == "--conf" and argv[i + 1].startswith("mode="):
            mode = argv[i + 1].split("=", 1)[1]
    print("READY 1.0.0 t", flush=True)
    for line in sys.stdin:
        cmd = line.split()
        if cmd[0] == "QUIT":
            break
        if cmd[0] == "RUN":
            print("OK 1 0.250000", flush=True)
        else:
            out = unquote(cmd[2]); os.makedirs(out)
            pq.write_table(pa.table({{"m": [mode]}}), out + "/part-0.parquet")
            print("OK 1 0.250000", flush=True)
    """
)


def make_runner(tmp_path: Path, name: str, build_info: dict | None = None) -> Path:
    binary = tmp_path / name
    binary.write_text(FAKE)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    if build_info is not None:
        Path(str(binary) + ".buildinfo").write_text(" ".join(f"{k}={v}" for k, v in build_info.items()))
    return binary


def q(sql: str = "select 1") -> Query:
    return Query("q01", sql, "tpch")


class Scripted:
    """An engine double: fixed rows, and a shared log of the timed runs in call order."""

    def __init__(self, side: str, rows: list[tuple], log: list[str], secs: float = 1.0):
        self.side, self.rows, self.log, self.secs = side, rows, log, secs

    def run(self, query):
        return self.rows

    def timed_run(self, query):
        self.log.append(self.side)
        return len(self.rows), self.secs


def test_abba_order_is_interleaved_per_block():
    log: list[str] = []
    entry = ab.one_query(q(), Scripted("a", [(1,)], log), Scripted("b", [(1,)], log, 2.0), reps=2)
    assert log == list("abbaabba")
    assert entry["status"] == "ok"
    assert entry["ratio"] == 2.0  # B / A on the scripted seconds
    assert len(entry["a_s"]) == len(entry["b_s"]) == 4
    assert entry["a_spread"] == 0.0


def test_mismatch_fails_the_query_and_is_never_timed():
    log: list[str] = []
    entry = ab.one_query(q(), Scripted("a", [(1,)], log), Scripted("b", [(2,)], log), reps=3)
    assert entry["status"] == "failed"
    assert "differ" in entry["reason"]
    assert log == []  # no timed run happened


def test_type_aware_match_and_ordered_comparison():
    log: list[str] = []
    ok = ab.one_query(q(), Scripted("a", [(1, 2.0)], log), Scripted("b", [(1, 2.0 + 1e-12)], log), reps=1)
    assert ok["status"] == "ok"
    rows_a, rows_b = [(1,), (2,)], [(2,), (1,)]
    unordered = ab.one_query(q(), Scripted("a", rows_a, log), Scripted("b", rows_b, log), reps=1)
    ordered = ab.one_query(q("select a from t order by a"), Scripted("a", rows_a, log), Scripted("b", rows_b, log), 1)
    assert unordered["status"] == "ok"
    assert ordered["status"] == "failed"


def test_conf_reaches_the_runner_and_different_conf_is_a_mismatch(tmp_path: Path):
    runner = make_runner(tmp_path, "runner")
    ds = Dataset("tpch", 1, tmp_path, ["t"])
    same = [KrishivEngine(binary=runner, conf=[("mode", "x"), ("k", "a=b")]) for _ in range(2)]
    diff = [KrishivEngine(binary=runner, conf=[("mode", "x")]), KrishivEngine(binary=runner, conf=[("mode", "y")])]
    try:
        for e in same + diff:
            e.timeout_s = 10
            e.setup(ds)
        assert ("k", "a=b") in same[0].conf
        assert ab.one_query(q(), same[0], same[1], reps=1)["status"] == "ok"
        entry = ab.one_query(q(), diff[0], diff[1], reps=1)
        assert entry["status"] == "failed"
        assert "differ" in entry["reason"]
    finally:
        for e in same + diff:
            e.teardown()


def test_conf_argv_exact(tmp_path: Path):
    runner = make_runner(tmp_path, "runner")
    eng = KrishivEngine(threads=3, memory_limit="1g", binary=runner, conf=[("a.b", "1"), ("c", "x=y")])
    try:
        eng.setup(Dataset("tpch", 1, tmp_path, ["t"]))
        argv = json.loads(Path(str(runner) + ".argv").read_text())
        assert argv[-4:] == ["--conf", "a.b=1", "--conf", "c=x=y"]
        assert eng.config()["conf"] == ["a.b=1", "c=x=y"]
    finally:
        eng.teardown()


def test_build_info_is_recorded_when_supported_and_skipped_otherwise(tmp_path: Path):
    old = KrishivEngine(binary=make_runner(tmp_path, "old"))
    new = KrishivEngine(binary=make_runner(tmp_path, "new", {"profile": "dist", "target-cpu": "x86-64-v3"}))
    assert old.runner_build_info() is None
    assert new.runner_build_info() == {"build_profile": "dist", "target_cpu": "x86-64-v3"}
    assert KrishivEngine(binary=tmp_path / "missing").runner_build_info() is None
    ds = Dataset("tpch", 1, tmp_path, ["t"])
    try:
        for e in (old, new):
            e.setup(ds)
        rec_old, rec_new = ab._side_record("A", old, None), ab._side_record("B", new, None)
        assert rec_old["build_info"] is None
        assert rec_old["git_sha"] is None
        assert rec_new["git_sha"] is None
        assert rec_new["build_info"] == {"build_profile": "dist", "target_cpu": "x86-64-v3"}
        assert ab._side_record("B", new, "explicit")["git_sha"] == "explicit"  # an explicit SHA wins
        assert rec_old["runner"] == str(old.binary)
        assert rec_old["version"] == "1.0.0"
    finally:
        for e in (old, new):
            e.teardown()


def test_parse_conf():
    assert ab.parse_conf(["a=1", "b=x=y", "c="]) == [("a", "1"), ("b", "x=y"), ("c", "")]
    assert ab.parse_conf(None) == []
    for bad in ("novalue", "=v"):
        with pytest.raises(ValueError, match="key=value"):
            ab.parse_conf([bad])


def test_cli_ab_arguments(monkeypatch: pytest.MonkeyPatch):
    seen: dict = {}
    monkeypatch.setattr(ab, "run_ab", lambda *a, **k: seen.update(args=a, kwargs=k))
    runners = ["--a-runner", "/m/r", "--b-runner", "/p/r"]
    confs = ["--a-conf", "x=1", "--b-conf", "x=2", "--b-conf", "y=3", "--a-sha", "aaa"]
    cli.main(["ab", "--suite", "tpch", "--sf", "10", *runners, *confs, "--reps", "2", "--queries", "q01,q05"])
    sf, dataset, a_run, b_run, a_conf, b_conf, reps, queries, *_rest = seen["args"]
    assert (sf, dataset, a_run, b_run) == (10.0, "tpch-duckdb", "/m/r", "/p/r")
    assert a_conf == [("x", "1")]
    assert b_conf == [("x", "2"), ("y", "3")]
    assert (reps, queries) == (2, ["q01", "q05"])
    assert seen["args"][-2:] == ("aaa", None)


def test_cli_ab_requires_both_runners_and_valid_conf(capsys):
    with pytest.raises(SystemExit):
        cli.main(["ab", "--a-runner", "/m/r"])
    with pytest.raises(SystemExit):
        cli.main(["ab", "--a-runner", "/a", "--b-runner", "/b", "--a-conf", "novalue"])
    assert "key=value" in capsys.readouterr().err


def test_runner_flag_wins_over_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", "/env/runner")
    assert runner_binary() == Path("/env/runner")
    assert runner_binary("/flag/runner") == Path("/flag/runner")
    assert KrishivEngine(binary="/flag/runner").binary == Path("/flag/runner")
    cli._set_runner("/cli/runner")  # the run/reproduce subcommands apply --runner through the env
    assert runner_binary() == Path("/cli/runner")
    cli._set_runner(None)
    assert runner_binary() == Path("/cli/runner")


def test_cli_runner_flag_is_accepted_by_every_subcommand(monkeypatch: pytest.MonkeyPatch):
    import krishiv_bench.abba as abba_mod

    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", "/env/runner")
    seen: dict = {}
    monkeypatch.setattr(abba_mod, "run_abba", lambda *a, **k: seen.update(args=a))
    cli.main(["abba", "--runner", "/flag/runner"])
    assert seen["args"][-1] == "/flag/runner"
    monkeypatch.setattr(cli, "run_suite", lambda *a, **k: Path("/x"))
    cli.main(["run", "--engine", "krishiv", "--runner", "/flag2/runner"])
    import os

    assert os.environ["KRISHIV_BENCH_RUNNER"] == "/flag2/runner"


def test_run_ab_writes_both_sides_into_the_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = tmp_path / "data"
    ds = data / "tpch-duckdb" / "sf1"
    ds.mkdir(parents=True)
    pq.write_table(pa.table({"a": [1]}), ds / "t.parquet")
    main_runner = make_runner(tmp_path, "main-runner", {"profile": "dist", "target-cpu": "x86-64-v3"})
    pr_runner = make_runner(tmp_path, "pr-runner")
    monkeypatch.setenv("KRISHIV_DATA", str(data))
    monkeypatch.setattr(ab, "load_queries", lambda *_: [q()])
    out = ab.run_ab(
        1,
        "tpch-duckdb",
        str(main_runner),
        str(pr_runner),
        [("mode", "x")],
        [("mode", "x")],
        reps=1,
        a_sha="mainsha",
        b_sha="prsha",
    )
    rec = json.loads(out.read_text())
    assert rec["kind"] == "ab"
    side_a, side_b = rec["sides"]["a"], rec["sides"]["b"]
    assert (side_a["runner"], side_a["git_sha"]) == (str(main_runner), "mainsha")
    assert side_a["build_info"] == {"build_profile": "dist", "target_cpu": "x86-64-v3"}
    assert (side_b["runner"], side_b["git_sha"], side_b["build_info"]) == (str(pr_runner), "prsha", None)
    assert side_a["conf"] == side_b["conf"] == ["mode=x"]
    assert rec["queries"]["q01"]["status"] == "ok"
    assert rec["summary"]["queries_ok"] == 1
    assert "B/A geomean" in ab.markdown(rec)
