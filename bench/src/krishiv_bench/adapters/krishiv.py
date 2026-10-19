"""Krishiv adapter: drives the bench-only ``krishiv-bench-runner`` binary (release build of ``krishiv-session``).

The Python facade cannot execute queries yet, so the adapter speaks the runner's line protocol
(see ``crates/krishiv-bench-runner/src/lib.rs``) over stdin/stdout: tables are registered once as Parquet
views at ``setup()`` (no pre-loading, for parity with the other adapters), then every query is one line.

* ``timed_run(query)`` -> ``(row_count, seconds)``: ``RUN``; the seconds are measured inside the runner
  around parse + analyse + plan + execute + materialising every batch.
* ``run(query)`` -> rows: ``DUMP`` (the result is written to Parquet outside the timed region, then read back),
  for correctness checks. The generic suite runner times this call, so use ``abba`` for Krishiv timings.

Build the runner first: ``cargo build --release -p krishiv-bench-runner`` (path override: ``$KRISHIV_BENCH_RUNNER``).
"""

from __future__ import annotations

import contextlib
import os
import select
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import Dataset, Engine, Query


class RunnerError(RuntimeError):
    """The runner answered ``ERR`` (the engine rejected or failed the query)."""


class RunnerCrashed(RuntimeError):
    """The runner died or timed out; it is restarted on the next command."""


def runner_binary() -> Path:
    env = os.environ.get("KRISHIV_BENCH_RUNNER")
    if env:
        return Path(env)
    target = Path(os.environ.get("CARGO_TARGET_DIR", Path(__file__).resolve().parents[4] / "target"))
    return target / "release" / "krishiv-bench-runner"


class KrishivEngine(Engine):
    name = "krishiv"

    def __init__(self, threads: int | None = None, memory_limit: str = "12g", binary: Path | None = None):
        self.threads = threads or os.cpu_count() or 1
        self.memory_limit = memory_limit
        self.binary = binary or runner_binary()
        self._proc: subprocess.Popen[str] | None = None
        self._dataset: Dataset | None = None
        self._version = "unknown"
        self._tmp = Path(tempfile.mkdtemp(prefix="krishiv-bench-"))
        self.timeout_s: float | None = None

    def version(self) -> str:
        return self._version

    def config(self) -> dict:
        return {"threads": self.threads, "memory_limit": self.memory_limit, "binary": self.binary.name}

    def build_info(self) -> dict[str, str]:
        return {"allocator": "mimalloc", "runner": "krishiv-bench-runner (release)"}

    # ------------------------------------------------------------------ process
    def _start(self) -> None:
        assert self._dataset is not None, "setup() not called"
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} missing: cargo build --release -p krishiv-bench-runner")
        cmd = [
            str(self.binary),
            "--data",
            str(self._dataset.root),
            "--threads",
            str(self.threads),
            "--memory-limit",
            self.memory_limit,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        ready = self._readline(120)
        parts = ready.split(" ", 2)
        if parts[0] != "READY" or len(parts) < 3:
            raise RunnerCrashed(f"runner did not start: {ready!r}")
        self._version = parts[1]

    def _readline(self, timeout: float | None) -> str:
        proc = self._proc
        assert proc is not None
        assert proc.stdout is not None
        if timeout is not None:
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            if not ready:
                self.cancel()
                raise RunnerCrashed(f"no answer within {timeout:.0f}s")
        line = proc.stdout.readline()
        if not line:
            self._kill()
            raise RunnerCrashed("runner exited")
        return line.rstrip("\n")

    def _command(self, text: str) -> str:
        if self._proc is None or self._proc.poll() is not None:
            self._kill()
            self._start()
        proc = self._proc
        assert proc is not None
        assert proc.stdin is not None
        try:
            proc.stdin.write(text + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            self._kill()
            raise RunnerCrashed("runner exited") from None
        reply = self._readline(self.timeout_s)
        if reply.startswith("ERR "):
            raise RunnerError(reply[4:])
        if not reply.startswith("OK "):
            raise RunnerCrashed(f"unexpected reply {reply!r}")
        return reply[3:]

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.kill()
            proc.wait()
            for f in (proc.stdin, proc.stdout):
                with contextlib.suppress(OSError):
                    if f:
                        f.close()

    def _sql_file(self, query: Query) -> Path:
        f = self._tmp / f"{query.id}.sql"
        f.write_text(query.sql)
        return f

    # ------------------------------------------------------------------ Engine
    def setup(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._start()

    def timed_run(self, query: Query) -> tuple[int, float]:
        """Executes to completion; returns the row count and the runner-measured seconds."""
        rows, secs = self._command(f"RUN {self._sql_file(query)}").split()
        return int(rows), float(secs)

    def run(self, query: Query) -> list[tuple]:
        import pyarrow.parquet as pq

        out = self._tmp / f"{query.id}.out"
        shutil.rmtree(out, ignore_errors=True)
        self._command(f"DUMP {self._sql_file(query)} {out}")
        parts = sorted(out.glob("*.parquet")) if out.is_dir() else [out]
        rows: list[tuple] = []
        for part in parts:
            table = pq.read_table(part)
            cols = [c.to_pylist() for c in table.columns]
            rows.extend(zip(*cols, strict=True) if cols else [])
        shutil.rmtree(out, ignore_errors=True)
        return rows

    def cancel(self) -> None:
        self._kill()

    def teardown(self) -> None:
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        self._kill()
        shutil.rmtree(self._tmp, ignore_errors=True)
