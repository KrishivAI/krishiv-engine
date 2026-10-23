#!/usr/bin/env python3
"""Crash triage for the cargo-fuzz targets of `fuzz/` (P2-T108).

A failing fuzz run leaves its reproducers in `fuzz/artifacts/<target>/` (`crash-*`, `timeout-*`, `oom-*`,
`leak-*`). `triage` shrinks each one with `cargo fuzz tmin` and stores it as a regression input,
`fuzz/regressions/<target>/<kind>-<sha1>`, named by content so a repeat finding is the same file. Commit those
files with the fix: every run (nightly, weekly, and `replay` locally) first replays the whole directory through
the real target with `-runs=0`, so a regression fails before any new input is tried.

Usage (from repo root):
    python3 tests/scripts/fuzz_triage.py triage <target> [--toolchain nightly-2026-08-12]
    python3 tests/scripts/fuzz_triage.py replay <target> [--toolchain nightly-2026-08-12]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

KINDS = ("crash", "timeout", "oom", "leak")
# libFuzzer limits shared with the nightly run (see .github/workflows/fuzz.yml).
LIMIT_ARGS = ("-timeout=10", "-rss_limit_mb=2048")
# A regression input above this size is kept (it still reproduces) but flagged: minimization did not help.
LARGE_BYTES = 256 * 1024

Runner = Callable[[Sequence[str]], int]


def _run(cmd: Sequence[str]) -> int:
    return subprocess.run(list(cmd), check=False).returncode


def host_triple(toolchain: str) -> str:
    """The host target of `toolchain` (`rustc -vV`).

    cargo-fuzz builds for the target it was itself compiled for unless told otherwise, and the prebuilt
    binaries CI installs are static musl executables: a sanitizer cannot link against musl's static libc, so
    every build must name the toolchain's host target.
    """
    out = subprocess.run(["rustc", f"+{toolchain}", "-vV"], check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ").strip()
    raise SystemExit(f"rustc +{toolchain} -vV printed no host line")


def reproducers(artifacts: Path) -> list[Path]:
    """The reproducer files of one target, in a stable order."""
    if not artifacts.is_dir():
        return []
    return sorted(p for p in artifacts.iterdir() if p.is_file() and p.name.split("-", 1)[0] in KINDS)


def regression_name(path: Path, data: bytes) -> str:
    """`<kind>-<sha1[:16]>` of the (minimized) content."""
    return f"{path.name.split('-', 1)[0]}-{hashlib.sha1(data, usedforsecurity=False).hexdigest()[:16]}"


def minimize(target: str, repro: Path, fuzz_dir: Path, toolchain: str, triple: str, run: Runner) -> Path:
    """Shrink `repro` with `cargo fuzz tmin`; fall back to the original when tmin cannot."""
    out = repro.with_name(f"minimized-{repro.name}")
    out.unlink(missing_ok=True)
    cmd = [
        "cargo",
        f"+{toolchain}",
        "fuzz",
        "tmin",
        "--target",
        triple,
        "--fuzz-dir",
        str(fuzz_dir),
        target,
        str(repro),
    ]
    code = run([*cmd, "--", "-runs=2000", *LIMIT_ARGS, f"-exact_artifact_path={out}"])
    return out if code == 0 and out.is_file() and out.stat().st_size > 0 else repro


def triage(target: str, fuzz_dir: Path, toolchain: str, triple: str, run: Runner = _run) -> list[Path]:
    """Store every reproducer of `target` as a minimized regression input; returns the files written."""
    written = []
    dest = fuzz_dir / "regressions" / target
    for repro in reproducers(fuzz_dir / "artifacts" / target):
        small = minimize(target, repro, fuzz_dir, toolchain, triple, run)
        data = small.read_bytes()
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / regression_name(repro, data)
        shutil.copyfile(small, out)
        written.append(out)
        note = " (large: minimization did not shrink it)" if len(data) > LARGE_BYTES else ""
        print(f"regression: {out} ({len(data)} bytes){note}")
    return written


def replay(target: str, fuzz_dir: Path, toolchain: str, triple: str, run: Runner = _run) -> int:
    """Run every committed regression input of `target` through the target once; nonzero when one fails."""
    dest = fuzz_dir / "regressions" / target
    if not dest.is_dir() or not any(dest.iterdir()):
        print(f"replay: no regression inputs for {target}")
        return 0
    cmd = ["cargo", f"+{toolchain}", "fuzz", "run", "--target", triple]
    cmd += ["--fuzz-dir", str(fuzz_dir), target, str(dest)]
    return run([*cmd, "--", "-runs=0", *LIMIT_ARGS])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("triage", "replay"))
    ap.add_argument("target")
    ap.add_argument("--toolchain", default="nightly-2026-08-12")
    ap.add_argument("--fuzz-dir", type=Path, default=Path("fuzz"))
    args = ap.parse_args(argv)
    triple = host_triple(args.toolchain)
    if args.command == "replay":
        return replay(args.target, args.fuzz_dir, args.toolchain, triple)
    found = triage(args.target, args.fuzz_dir, args.toolchain, triple)
    if not found:
        print(f"triage: no reproducers under {args.fuzz_dir / 'artifacts' / args.target}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
