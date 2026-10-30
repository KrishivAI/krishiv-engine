"""Bisect: the search, commit listing, the build step, both probes and the run (fake runner, real git)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_ab import make_runner

from krishiv_bench import bisect
from krishiv_bench.adapters.base import Dataset, Query
from krishiv_bench.bisect import BisectError, Builder, first_bad


def search(n: int, bad_from: int, unjudgeable: frozenset[int] | set[int] = frozenset()):
    commits = [f"c{i}" for i in range(n)]
    probed: list[str] = []

    def probe(sha: str):
        probed.append(sha)
        i = int(sha[1:])
        return None if i in unjudgeable else i >= bad_from

    culprits, verdicts = first_bad(commits, probe)
    return culprits, verdicts, probed


@pytest.mark.parametrize("n", [1, 2, 7, 16])
def test_first_bad_exact_at_every_position(n):
    for bad_from in range(n):
        culprits, _verdicts, probed = search(n, bad_from)
        assert culprits == [f"c{bad_from}"]
        assert len(probed) == len(set(probed))  # no commit is built twice
        assert len(probed) <= max(1, n.bit_length())


def test_first_bad_empty_range_is_an_error():
    with pytest.raises(BisectError, match="no commits"):
        first_bad([], lambda s: True)


def test_first_bad_skips_unjudgeable_commits():
    culprits, _, _ = search(10, 6, unjudgeable={4, 5})
    assert culprits == ["c4", "c5", "c6"]  # c4 and c5 might be the culprit: they cannot be built
    culprits, _, _ = search(10, 6, unjudgeable={6})
    assert culprits == ["c6", "c7"]


def test_first_bad_reports_the_range_when_the_boundary_is_unjudgeable():
    culprits, verdicts, _ = search(6, 3, unjudgeable={2, 3})
    assert culprits == ["c2", "c3", "c4"]
    assert verdicts["c4"] is True


def test_first_bad_everything_unjudgeable_returns_all_but_known_bad():
    culprits, _, _ = search(4, 0, unjudgeable={0, 1, 2})
    assert culprits == ["c0", "c1", "c2", "c3"]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.name", "T")
    git(r, "config", "user.email", "t@example.com")
    for i in range(4):
        (r / "f").write_text(str(i))
        git(r, "add", "f")
        git(r, "commit", "-q", "-m", f"commit {i}")
    return r


def test_commits_between_is_oldest_first_and_excludes_good(repo):
    shas = git(repo, "rev-list", "--reverse", "HEAD").split()
    assert bisect.commits_between(repo, shas[0], "HEAD") == shas[1:]
    assert bisect.commits_between(repo, shas[2], shas[3]) == [shas[3]]
    assert bisect.commits_between(repo, shas[3], shas[3]) == []


def test_commits_between_rejects_unknown_and_non_ancestor(repo):
    shas = git(repo, "rev-list", "--reverse", "HEAD").split()
    with pytest.raises(BisectError, match="unknown revision"):
        bisect.commits_between(repo, "nope", "HEAD")
    with pytest.raises(BisectError, match="not on the first-parent history"):
        bisect.commits_between(repo, shas[3], shas[0])


def test_commits_between_rejects_a_good_commit_on_a_merged_branch(repo):
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "s").write_text("1")
    git(repo, "add", "s")
    git(repo, "commit", "-q", "-m", "side work")
    side = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    (repo / "m").write_text("1")
    git(repo, "add", "m")
    git(repo, "commit", "-q", "-m", "main work")
    git(repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    merge = git(repo, "rev-parse", "HEAD")
    assert len(bisect.commits_between(repo, base, merge)) == 2  # main work, then the merge: side commits are folded in
    with pytest.raises(BisectError, match="first-parent"):
        bisect.commits_between(repo, side, merge)


def test_describe_commit(repo):
    sha = git(repo, "rev-parse", "HEAD")
    info = bisect.describe_commit(repo, sha)
    assert info["sha"] == sha
    assert info["subject"] == "commit 3"
    assert info["author"] == "T"


class FakeRun:
    """Stands in for ``subprocess.run``: git calls pass through, cargo "builds" a file unless told to fail."""

    def __init__(self, target: Path, fail: frozenset[str] | set[str] = frozenset(), source: Path | None = None):
        self.target, self.fail, self.source = target, fail, source
        self.builds: list[str] = []

    def __call__(self, cmd, **kw):
        if cmd[0] != "cargo":
            return subprocess.run(cmd, **{k: v for k, v in kw.items() if k != "timeout"})
        sha = git_head(kw["cwd"])
        self.builds.append(sha)
        if sha in self.fail:
            return subprocess.CompletedProcess(cmd, 101, "", "error: boom\n")
        out = self.target / "dist"
        out.mkdir(parents=True, exist_ok=True)
        (out / bisect.RUNNER_CRATE).write_bytes((self.source or Path()).read_bytes())
        return subprocess.CompletedProcess(cmd, 0, "", "")


def git_head(cwd) -> str:
    return git(Path(cwd), "rev-parse", "HEAD")


def test_builder_builds_once_per_commit_and_caches(repo, tmp_path):
    fake = make_runner(tmp_path, "fake")
    work = tmp_path / "work"
    run = FakeRun(work / "target", source=fake)
    b = Builder(repo, work, run=run)
    head = git(repo, "rev-parse", "HEAD")
    first = b.runner(head)
    assert first is not None
    assert first.is_file()
    assert b.runner(head) == first
    assert run.builds == [head]
    prev = git(repo, "rev-parse", "HEAD~1")
    assert b.runner(prev) is not None  # the scratch worktree is moved, not re-created
    assert run.builds == [head, prev]
    b.close()
    assert not b.tree.exists()
    assert not (work / "target").exists()


def test_builder_returns_none_when_the_build_fails(repo, tmp_path, capsys):
    head = git(repo, "rev-parse", "HEAD")
    work = tmp_path / "work"
    b = Builder(repo, work, run=FakeRun(work / "target", fail={head}, source=make_runner(tmp_path, "fake")))
    assert b.runner(head) is None
    assert "build failed" in capsys.readouterr().out
    b.close()


class StubBuilder:
    """Maps a commit to a runner path; None for a commit that does not build."""

    def __init__(self, runners: dict[str, Path | None]):
        self.runners, self.closed = runners, False

    def runner(self, sha):
        return self.runners.get(sha)

    def close(self):
        self.closed = True


def make_ds(tmp_path: Path) -> Dataset:
    return Dataset("fake", 1.0, tmp_path, ["t"])


def test_correctness_probe_good_bad_error_and_unbuildable(tmp_path):
    good = make_runner(tmp_path, "good")
    ds, query = make_ds(tmp_path), Query("q01", "select 1", "tpch")
    builder = StubBuilder({"g": good, "b": good, "n": None})
    probe = bisect.correctness_probe(builder, ds, query, [("default",)], 1, "1g", 30)
    assert probe("g") is False
    assert probe("n") is None
    wrong = bisect.correctness_probe(builder, ds, query, [("other",)], 1, "1g", 30)
    assert wrong("b") is True


def test_correctness_probe_start_failure_is_skipped_and_query_error_is_bad(tmp_path):
    crash = tmp_path / "crash"
    crash.write_text("#!/bin/sh\nexit 3\n")
    crash.chmod(0o755)
    failing = tmp_path / "failing"
    failing.write_text("#!/bin/sh\necho 'READY 1.0.0 t'\nwhile read l; do echo 'ERR boom'; done\n")
    failing.chmod(0o755)
    probe = bisect.correctness_probe(
        StubBuilder({"c": crash, "f": failing}), make_ds(tmp_path), Query("q01", "select 1", "tpch"), [], 1, "1g", 5
    )
    assert probe("c") is None  # never reached READY: not evidence about the commit
    assert probe("f") is True


def test_time_probe_flags_a_failure_and_judges_by_ratio(tmp_path):
    good = make_runner(tmp_path, "good")
    ds, query = make_ds(tmp_path), Query("q01", "select 1", "tpch")
    settled = []
    builder = StubBuilder({"same": good, "n": None})
    probe = bisect.time_probe(builder, ds, query, good, 1, "1g", 30, 1, 1.10, settle=lambda: settled.append(1))
    # The fake runner reports 0.25 s always: ratio 1, never bad.
    assert probe("same") is False
    assert probe("n") is None
    assert settled == [1]
    # A probed build whose runner exits at start cannot be judged; one that fails the query is bad.
    crash = tmp_path / "crash"
    crash.write_text("#!/bin/sh\nexit 3\n")
    crash.chmod(0o755)
    builder.runners["c"] = crash
    assert probe("c") is None
    failing = tmp_path / "failing"
    failing.write_text("#!/bin/sh\necho 'READY 1.0.0 t'\nwhile read l; do echo 'ERR boom'; done\n")
    failing.chmod(0o755)
    builder.runners["f"] = failing
    assert probe("f") is True


def test_time_probe_a_side_failure_is_not_blamed_on_the_commit(tmp_path, monkeypatch):
    good = make_runner(tmp_path, "good")
    monkeypatch.setattr(bisect, "one_query", lambda *a: {"status": "failed", "reason": "a crashed/timed out: x"})
    probe = bisect.time_probe(
        StubBuilder({"x": good}), make_ds(tmp_path), Query("q01", "select 1", "tpch"), good, 1, "1g", 30, 1, 1.10,
        settle=lambda: None,
    )  # fmt: skip
    assert probe("x") is None
    monkeypatch.setattr(bisect, "one_query", lambda *a: {"status": "failed", "reason": "b error: boom"})
    assert probe("x") is True


def test_builder_timeout_and_checkout_failure_skip_the_commit(repo, tmp_path, capsys):
    head = git(repo, "rev-parse", "HEAD")
    work = tmp_path / "work"

    def timing_out(cmd, **kw):
        if cmd[0] == "cargo":
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.run(cmd, **{k: v for k, v in kw.items() if k != "timeout"})

    assert Builder(repo, work, run=timing_out).runner(head) is None
    assert "timed out" in capsys.readouterr().out
    assert Builder(repo, tmp_path / "w2").runner("0" * 40) is None  # git refuses the checkout
    assert "checkout failed" in capsys.readouterr().out


def test_builder_close_removes_the_cached_binaries(repo, tmp_path):
    work = tmp_path / "work"
    b = Builder(repo, work, run=FakeRun(work / "target", source=make_runner(tmp_path, "fake")))
    assert b.runner(git(repo, "rev-parse", "HEAD")) is not None
    b.close()
    assert not b.bin.exists()


def test_time_probe_ratio_threshold(tmp_path, monkeypatch):
    entries = iter(
        [
            {"status": "ok", "ratio": 1.5, "ratio_ci95": [1.2, 1.8]},
            {"status": "ok", "ratio": 1.5, "ratio_ci95": [0.9, 1.8]},  # CI includes 1: noise
            {"status": "ok", "ratio": 1.05, "ratio_ci95": [1.02, 1.08]},  # below the threshold
        ]
    )
    monkeypatch.setattr(bisect, "one_query", lambda *a, **k: next(entries))
    good = make_runner(tmp_path, "good")
    probe = bisect.time_probe(
        StubBuilder({"x": good}), make_ds(tmp_path), Query("q01", "select 1", "tpch"), good, 1, "1g", 30, 1, 1.10,
        settle=lambda: None,
    )  # fmt: skip
    assert [probe("x"), probe("x"), probe("x")] == [True, False, False]


def test_render_exact_and_range():
    base = {"suite": "tpch", "query": "q01", "mode": "time", "good": "a", "bad": "b", "commits": 5, "probes": 3}
    c = {"sha": "a" * 40, "author": "T", "date": "2026-10-30", "subject": "slow it down"}
    one = bisect.render({**base, "culprits": [c]})
    assert "First bad commit: aaaaaaaaaaaa" in one
    many = bisect.render({**base, "culprits": [c, c]})
    assert "one of these 2" in many


def run_with(monkeypatch, tmp_path, repo, mode, builder, **kw):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(bisect, "resolve_dataset", lambda *a, **k: make_ds(tmp_path))
    monkeypatch.setattr(bisect, "load_queries", lambda suite, eng, only: [Query("q01", "select 1", suite)])

    class Duck:
        def __init__(self, **k):
            pass

        def setup(self, ds):
            pass

        def run(self, q):
            return [("default",)]

        def teardown(self):
            pass

    monkeypatch.setattr(bisect, "DuckDBEngine", Duck)
    shas = git(repo, "rev-list", "--reverse", "HEAD").split()
    return bisect.run_bisect(
        "tpch", "q01", shas[0], shas[-1], mode, repo=repo, work=tmp_path / "w", builder=builder, **kw
    ), shas


def test_run_bisect_correctness_finds_the_first_bad_commit(tmp_path, repo, monkeypatch):
    good, bad = make_runner(tmp_path, "good"), make_runner(tmp_path, "bad")
    # make the "bad" runner answer differently via its mode conf: patch the engine rows instead
    shas = git(repo, "rev-list", "--reverse", "HEAD").split()
    builder = StubBuilder({shas[1]: good, shas[2]: bad, shas[3]: bad})
    real = bisect.KrishivEngine

    class Eng(real):
        def run(self, query):
            return [("wrong",)] if self.binary.name == "bad" else [("default",)]

    monkeypatch.setattr(bisect, "KrishivEngine", Eng)
    f, _ = run_with(monkeypatch, tmp_path, repo, "correctness", builder)
    rec = json.loads(f.read_text())
    assert [c["sha"] for c in rec["culprits"]] == [shas[2]]
    assert rec["commits"] == 3
    assert builder.closed


def test_run_bisect_refuses_when_the_bad_end_is_fine(tmp_path, repo, monkeypatch):
    good = make_runner(tmp_path, "good")
    shas = git(repo, "rev-list", "--reverse", "HEAD").split()
    builder = StubBuilder(dict.fromkeys(shas, good))
    with pytest.raises(BisectError, match="does not show the problem"):
        run_with(monkeypatch, tmp_path, repo, "correctness", builder)
    assert builder.closed


def test_run_bisect_unjudgeable_bad_end_is_not_reported_as_fine(tmp_path, repo, monkeypatch):
    builder = StubBuilder({})  # nothing builds
    with pytest.raises(BisectError, match="could not be judged"):
        run_with(monkeypatch, tmp_path, repo, "correctness", builder)


def test_run_bisect_time_mode_needs_a_buildable_good_commit(tmp_path, repo, monkeypatch):
    builder = StubBuilder({})
    with pytest.raises(BisectError, match="does not build"):
        run_with(monkeypatch, tmp_path, repo, "time", builder)


def test_run_bisect_time_mode_end_to_end(tmp_path, repo, monkeypatch):
    good = make_runner(tmp_path, "good")
    shas = git(repo, "rev-list", "--reverse", "HEAD").split()
    builder = StubBuilder(dict.fromkeys(shas, good))
    slow = {shas[2], shas[3]}

    def fake_one_query(q, a, b, reps):
        bad = Path(b.binary).name == "slow"
        return {"status": "ok", "ratio": 2.0 if bad else 1.0, "ratio_ci95": [1.5, 2.5] if bad else [0.9, 1.1]}

    slow_runner = make_runner(tmp_path, "slow")
    builder.runners.update({s: slow_runner for s in slow})
    monkeypatch.setattr(bisect, "one_query", fake_one_query)
    f, _ = run_with(monkeypatch, tmp_path, repo, "time", builder)
    assert [c["sha"] for c in json.loads(f.read_text())["culprits"]] == [shas[2]]


def test_run_bisect_empty_range_and_bad_arguments(tmp_path, repo, monkeypatch):
    shas = git(repo, "rev-list", "--reverse", "HEAD").split()
    builder = StubBuilder({})
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(bisect, "resolve_dataset", lambda *a, **k: make_ds(tmp_path))
    monkeypatch.setattr(bisect, "load_queries", lambda suite, eng, only: [Query("q01", "select 1", suite)])
    with pytest.raises(BisectError, match="no commits between"):
        bisect.run_bisect("tpch", "q01", shas[1], shas[1], repo=repo, builder=builder, work=tmp_path / "w")
    with pytest.raises(BisectError, match="--reps"):
        bisect.run_bisect("tpch", "q01", shas[0], shas[1], repo=repo, reps=0)
    with pytest.raises(BisectError, match="--threshold"):
        bisect.run_bisect("tpch", "q01", shas[0], shas[1], repo=repo, threshold=0)


def test_resolve_rejects_option_like_revisions(repo):
    with pytest.raises(BisectError, match="unknown revision"):
        bisect.resolve(repo, "--all")


def test_run_bisect_rejects_a_bad_mode_and_unknown_query(tmp_path, repo, monkeypatch):
    with pytest.raises(BisectError, match="mode"):
        bisect.run_bisect("tpch", "q01", "a", "b", "fast", repo=repo)
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    monkeypatch.setattr(bisect, "resolve_dataset", lambda *a, **k: make_ds(tmp_path))
    monkeypatch.setattr(bisect, "load_queries", lambda *a: [])
    with pytest.raises(SystemExit, match="unknown query"):
        bisect.run_bisect("tpch", "q99", "a", "b", repo=repo)
