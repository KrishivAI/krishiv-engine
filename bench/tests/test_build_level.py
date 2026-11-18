"""The shipped CPU level: a timed run refuses a runner built for another one (study 28: 1.14x over TPC-H)."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_ab import make_runner
from test_bisect import FakeRun, git

from krishiv_bench import reproduce
from krishiv_bench.adapters.base import Dataset
from krishiv_bench.adapters.krishiv import (
    ANY_BUILD_ENV,
    KrishivEngine,
    RunnerBuildMismatch,
    build_level_problem,
    read_build_info,
    require_shipped_build,
    shipped_rustflags,
)
from krishiv_bench.bisect import Builder

DEFAULT = {"profile": "dist", "target-cpu": "default", "branch-align": "off"}
SHIPPED = {"profile": "dist", "target-cpu": "x86-64-v3", "branch-align": "32B"}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository of one commit, for the bisect builder."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "-c", "user.name=T", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "c0")
    return r


@pytest.fixture(autouse=True)
def _x86(monkeypatch):
    """Judged as on the laptop (x86_64), whatever machine runs the tests."""
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.delenv(ANY_BUILD_ENV, raising=False)


def test_problem_names_the_level_and_the_fix(tmp_path):
    p = build_level_problem(tmp_path / "r", {"build_profile": "dist", "target_cpu": "default"})
    assert p is not None
    assert "target-cpu=default" in p
    assert "x86-64-v3" in p
    assert ANY_BUILD_ENV in p


def test_problem_on_arm_names_the_arm_build(tmp_path):
    p = build_level_problem(tmp_path / "r", {"target_cpu": "default"}, "aarch64")
    assert p is not None
    assert "RUSTFLAGS='-C target-cpu=neoverse-n1' cargo build" in p


@pytest.mark.parametrize(
    ("cpu", "machine", "judged"),
    [
        ("x86-64-v3", "x86_64", False),
        ("unknown", "x86_64", False),  # an old runner or a test double cannot say
        ("default", "x86_64", True),
        ("x86-64-v4", "x86_64", True),
        ("default", "aarch64", True),
        ("neoverse-n1", "aarch64", False),
        ("default", "riscv64", False),  # no shipped level there
    ],
)
def test_only_a_known_other_level_is_a_problem(tmp_path, cpu, machine, judged):
    assert (build_level_problem(tmp_path / "r", {"target_cpu": cpu}, machine) is not None) == judged


def test_shipped_rustflags_per_machine():
    assert shipped_rustflags("x86_64") == "-C target-cpu=x86-64-v3 -C llvm-args=-x86-branches-within-32B-boundaries"
    assert shipped_rustflags("aarch64") == "-C target-cpu=neoverse-n1"
    assert shipped_rustflags("riscv64") == ""


def test_read_build_info_of_a_runner_and_of_one_that_cannot_say(tmp_path):
    assert read_build_info(make_runner(tmp_path, "new", SHIPPED)) == {
        "build_profile": "dist",
        "target_cpu": "x86-64-v3",
        "branch_align": "32B",
    }
    assert read_build_info(make_runner(tmp_path, "old")) == {"build_profile": "unknown", "target_cpu": "unknown"}
    assert read_build_info(tmp_path / "missing")["target_cpu"] == "unknown"


def test_setup_refuses_a_default_runner_before_starting_it(tmp_path):
    binary = make_runner(tmp_path, "def", DEFAULT)
    eng = KrishivEngine(binary=binary)
    try:
        with pytest.raises(RunnerBuildMismatch, match="target-cpu=default"):
            eng.setup(Dataset("tpch", 1, tmp_path, ["t"]))
        assert not (tmp_path / "def.argv").exists()  # never started
    finally:
        eng.teardown()


@pytest.mark.parametrize("info", [SHIPPED, None])
def test_setup_starts_a_shipped_or_unreporting_runner(tmp_path, info):
    eng = KrishivEngine(binary=make_runner(tmp_path, "ok", info))
    try:
        eng.setup(Dataset("tpch", 1, tmp_path, ["t"]))
        assert (tmp_path / "ok.argv").exists()
    finally:
        eng.teardown()


def test_any_build_env_lets_a_default_runner_run(tmp_path, monkeypatch):
    monkeypatch.setenv(ANY_BUILD_ENV, "1")
    eng = KrishivEngine(binary=make_runner(tmp_path, "def", DEFAULT))
    try:
        eng.setup(Dataset("tpch", 1, tmp_path, ["t"]))
        assert (tmp_path / "def.argv").exists()
    finally:
        eng.teardown()


def test_any_build_env_zero_does_not(tmp_path, monkeypatch):
    monkeypatch.setenv(ANY_BUILD_ENV, "0")
    with pytest.raises(SystemExit, match="target-cpu=default"):
        require_shipped_build(make_runner(tmp_path, "def", DEFAULT))


def test_require_shipped_build_resolves_the_env_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(make_runner(tmp_path, "def", DEFAULT)))
    with pytest.raises(SystemExit, match="target-cpu=default"):
        require_shipped_build()
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(make_runner(tmp_path, "v3", SHIPPED)))
    require_shipped_build()
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(tmp_path / "missing"))
    require_shipped_build()  # a missing runner is reported where it is started


def test_reproduce_refuses_before_taking_the_bench_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(make_runner(tmp_path, "def", DEFAULT)))
    monkeypatch.setattr(reproduce, "resolve_dataset", lambda *a, **k: pytest.fail("ran past the build check"))
    with pytest.raises(SystemExit, match="target-cpu=default"):
        reproduce.run_reproduce("tpch", 1, engines=["krishiv", "duckdb"])


def test_reproduce_without_krishiv_does_not_ask(tmp_path, monkeypatch):
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(make_runner(tmp_path, "def", DEFAULT)))

    class Stop(Exception):
        pass

    def stop(*_a, **_k):
        raise Stop

    monkeypatch.setattr(reproduce, "resolve_dataset", stop)
    with pytest.raises(Stop):
        reproduce.run_reproduce("tpch", 1, engines=["duckdb", "polars"])


def test_bisect_builds_at_the_shipped_level_after_the_callers_flags(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("RUSTFLAGS", "-C linker=clang")
    work = tmp_path / "work"
    run = FakeRun(work / "target", source=make_runner(tmp_path, "fake"))
    b = Builder(repo, work, run=run)
    assert b.runner(git(repo, "rev-parse", "HEAD")) is not None
    assert run.envs[0]["RUSTFLAGS"] == (
        "-C linker=clang -C target-cpu=x86-64-v3 -C llvm-args=-x86-branches-within-32B-boundaries"
    )


def test_bisect_without_caller_flags(repo, tmp_path, monkeypatch):
    monkeypatch.delenv("RUSTFLAGS", raising=False)
    work = tmp_path / "work"
    run = FakeRun(work / "target", source=make_runner(tmp_path, "fake"))
    assert Builder(repo, work, run=run).runner(git(repo, "rev-parse", "HEAD")) is not None
    assert run.envs[0]["RUSTFLAGS"].startswith("-C target-cpu=x86-64-v3")
