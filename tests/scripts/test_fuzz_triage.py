"""Unit tests for fuzz_triage.py (no cargo-fuzz needed: the command runner is faked).

Run: (cd bench && uv run --group dev pytest ../tests/scripts/test_fuzz_triage.py)
"""

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fuzz_triage as f

TRIPLE = "x86_64-unknown-linux-gnu"


def _artifacts(tmp_path: Path) -> Path:
    art = tmp_path / "artifacts" / "t"
    art.mkdir(parents=True)
    (art / "crash-aaaa").write_bytes(b"KRSHxxxxxxxx")
    (art / "timeout-bbbb").write_bytes(b"slow")
    (art / "README").write_text("not a reproducer")
    return art


def test_reproducers_picks_only_known_kinds(tmp_path):
    art = _artifacts(tmp_path)
    assert [p.name for p in f.reproducers(art)] == ["crash-aaaa", "timeout-bbbb"]
    assert f.reproducers(tmp_path / "missing") == []


def test_triage_stores_minimized_input_named_by_content(tmp_path):
    _artifacts(tmp_path)
    calls: list[Sequence[str]] = []

    def fake_tmin(cmd: Sequence[str]) -> int:
        calls.append(cmd)
        out = next(c for c in cmd if c.startswith("-exact_artifact_path=")).split("=", 1)[1]
        Path(out).write_bytes(b"KRSH")
        return 0

    written = f.triage("t", tmp_path, "nightly-x", TRIPLE, fake_tmin)
    assert len(calls) == 2
    assert all(c[:3] == ["cargo", "+nightly-x", "fuzz"] for c in calls)
    # Every build names the host target (see host_triple).
    assert all(c[c.index("--target") + 1] == TRIPLE for c in calls)
    assert {w.parent for w in written} == {tmp_path / "regressions" / "t"}
    assert all(w.read_bytes() == b"KRSH" for w in written)
    assert written[0].name.startswith("crash-")
    assert written[1].name.startswith("timeout-")
    # Same content, same kind: the same name, so a repeat finding does not add a file.
    assert f.triage("t", tmp_path, "nightly-x", TRIPLE, fake_tmin) == written


def test_triage_keeps_original_when_tmin_fails(tmp_path):
    _artifacts(tmp_path)
    written = f.triage("t", tmp_path, "nightly-x", TRIPLE, lambda _cmd: 1)
    assert sorted(w.read_bytes() for w in written) == [b"KRSHxxxxxxxx", b"slow"]


def test_replay_skips_without_regressions_and_passes_the_exit_code(tmp_path):
    assert f.replay("t", tmp_path, "nightly-x", TRIPLE, lambda _cmd: 99) == 0
    dest = tmp_path / "regressions" / "t"
    dest.mkdir(parents=True)
    (dest / "crash-1").write_bytes(b"x")
    seen: list[Sequence[str]] = []

    def fake_run(cmd: Sequence[str]) -> int:
        seen.append(cmd)
        return 77

    assert f.replay("t", tmp_path, "nightly-x", TRIPLE, fake_run) == 77
    assert "-runs=0" in seen[0]
    assert seen[0][seen[0].index("--target") + 1] == TRIPLE
    assert str(dest) in seen[0]


def test_main_triage_fails_without_reproducers(tmp_path, capsys):
    assert f.main(["triage", "t", "--fuzz-dir", str(tmp_path)]) == 1
    assert "no reproducers" in capsys.readouterr().out


def test_host_triple_reads_the_toolchains_host_line(monkeypatch):
    class Done:
        stdout = "rustc 1.99.0-nightly\nbinary: rustc\nhost: aarch64-apple-darwin\nrelease: 1.99.0-nightly\n"

    seen: list[Sequence[str]] = []

    def fake(cmd, **_kw):
        seen.append(cmd)
        return Done()

    monkeypatch.setattr(f.subprocess, "run", fake)
    assert f.host_triple("nightly-x") == "aarch64-apple-darwin"
    assert seen == [["rustc", "+nightly-x", "-vV"]]
    Done.stdout = "rustc 1.99.0\n"
    with pytest.raises(SystemExit, match="no host line"):
        f.host_triple("nightly-x")


def test_main_passes_the_host_triple(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "host_triple", lambda _tc: TRIPLE)
    dest = tmp_path / "regressions" / "t"
    dest.mkdir(parents=True)
    (dest / "crash-1").write_bytes(b"x")
    seen: list[Sequence[str]] = []
    # The default runner shells out through subprocess.run.
    monkeypatch.setattr(f.subprocess, "run", lambda cmd, **_kw: seen.append(cmd) or type("R", (), {"returncode": 0})())
    assert f.main(["replay", "t", "--fuzz-dir", str(tmp_path)]) == 0
    assert seen[0][seen[0].index("--target") + 1] == TRIPLE
