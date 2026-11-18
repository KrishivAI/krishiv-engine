# SPDX-License-Identifier: Apache-2.0
"""`pytest --collect-only` of a pinned checkout: the collected denominator and its drift check."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import Entry

COLLECT_TIMEOUT_S = 15 * 60


class CollectError(RuntimeError):
    """pytest could not collect (import error, no tests): never fall back to the static census silently."""


def suite_env(checkout: Path, scratch: Path, port: int | None = None) -> dict[str, str]:
    """Environment for pytest on the suite: `pyspark` from the checkout, a stub `SPARK_HOME`.

    `pyspark.testing.sqlutils` calls `_find_spark_home()` at import, which wants `bin/spark-submit` and a `jars`
    directory. A Python-only sparse checkout has neither, and a real Spark home would let tests start a JVM, so the
    stub is always an empty one.
    """
    home = scratch / "spark-home"
    (home / "bin").mkdir(parents=True, exist_ok=True)
    (home / "jars").mkdir(exist_ok=True)
    (home / "bin" / "spark-submit").touch()
    env = {k: v for k, v in os.environ.items() if k != "SPARK_HOME"}
    env["SPARK_HOME"] = str(home)
    env["PYTHONPATH"] = str(checkout / "python")
    if port is not None:
        env["SPARK_CONNECT_TESTING_REMOTE"] = f"sc://127.0.0.1:{port}"
    return env


def parse_collected(stdout: str, connect_dir: Path) -> set[str]:
    """Node ids (`file::Class::test`, relative to `connect_dir`) from `pytest --collect-only -q` output."""
    out: set[str] = set()
    for line in stdout.splitlines():
        parts = line.strip().split("::")
        if len(parts) != 3 or not parts[0].endswith(".py"):
            continue
        path = Path(parts[0])
        try:
            rel = (path.relative_to(connect_dir) if path.is_absolute() else path).as_posix()
        except ValueError:
            continue
        out.add(f"{rel}::{parts[1]}::{parts[2]}")
    return out


def collect_ids(checkout: Path) -> set[str]:
    """Run `pytest --collect-only` over the connect tests of `checkout` with this interpreter."""
    connect_dir = checkout / "python" / "pyspark" / "sql" / "tests" / "connect"
    with tempfile.TemporaryDirectory(prefix="pyspark-collect-") as scratch:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(connect_dir),
            "--collect-only",
            "-q",
            f"--rootdir={connect_dir}",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
        ]
        try:
            proc = subprocess.run(
                cmd,
                env=suite_env(checkout, Path(scratch)),
                capture_output=True,
                text=True,
                timeout=COLLECT_TIMEOUT_S,
                check=False,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired as e:
            raise CollectError(f"pytest --collect-only timed out after {COLLECT_TIMEOUT_S}s") from e
    ids = parse_collected(proc.stdout, connect_dir)
    # Exit 0 = collected; 2 = interrupted or collection errors (a partial count would be a silent drift).
    if proc.returncode != 0 or not ids:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        raise CollectError(f"pytest --collect-only exited {proc.returncode} with {len(ids)} ids:\n{tail}")
    return ids


@dataclass
class Drift:
    """Differences between the pytest collection and what the committed summary and manifest assume."""

    version: str
    collected: int
    summary_collected: int | None
    unmatched_entries: list[str] = field(default_factory=list)
    census_only: list[str] = field(default_factory=list)  # in the static census, not collected by pytest
    pytest_only: list[str] = field(default_factory=list)  # collected by pytest, not in the static census

    @property
    def drifted(self) -> bool:
        return self.collected != self.summary_collected or bool(self.unmatched_entries)

    def lines(self) -> list[str]:
        out = [f"{self.version}: pytest collected {self.collected}, committed summary has {self.summary_collected}"]
        out += [f"  manifest entry matches no collected test: {e}" for e in self.unmatched_entries]
        for label, ids in (("census only", self.census_only), ("pytest only", self.pytest_only)):
            if ids:
                out.append(f"  {len(ids)} {label}, first: {', '.join(ids[:5])}")
        return out


def check_drift(
    version: str, collected: set[str], census_ids: set[str], summary: dict | None, entries: list[Entry]
) -> Drift:
    """Compare the collection with the committed summary's count and the manifest's entries."""
    prev = summary["denominator"]["collected"] if summary else None
    unmatched = [e.target for e in entries if e.applies(version) and not any(e.matches(t) for t in collected)]
    return Drift(
        version,
        len(collected),
        prev,
        unmatched,
        sorted(census_ids - collected),
        sorted(collected - census_ids),
    )
