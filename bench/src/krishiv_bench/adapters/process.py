"""Parent side of ``krishiv_bench.worker``: an Engine whose adapter lives in a child process session."""

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess
import sys
from typing import Any

from .. import worker
from .base import Dataset, Engine, Measure, Query
from .proc_mem import TreeProc


class EngineError(RuntimeError):
    """The engine raised while running the request (the worker stays usable)."""


class EngineTimeout(RuntimeError):
    """The request exceeded its deadline; the worker's process group was killed and restarts on the next request."""


class EngineCrashed(RuntimeError):
    """The worker died (killed by the kernel, OOM, segfault); it restarts on the next request."""


class ProcessEngine(Engine):
    def __init__(self, name: str, threads: int, memory_gb: int, timeout_s: float | None = None):
        self.name = name
        self.threads = threads
        self.memory_gb = memory_gb
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._dataset: Dataset | None = None
        self.info: dict[str, Any] = {}
        self._tree: TreeProc | None = None

    # -- process management -------------------------------------------------
    def _start(self) -> None:
        assert self._dataset is not None, "setup() not called"
        # -I: ignore PYTHONPATH and the cwd; krishiv_bench is importable because it is installed in the venv.
        self._proc = subprocess.Popen(
            [sys.executable, "-I", "-m", "krishiv_bench.worker", self.name, str(self.threads), str(self.memory_gb)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            self.info = self._call({"op": "setup", "dataset": self._dataset}, self.setup_timeout_s)
        except BaseException:
            self.kill()  # a worker that failed setup has no tables; the next request must start a clean one
            raise

    setup_timeout_s = 900.0

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        for f in (proc.stdin, proc.stdout):
            with contextlib.suppress(OSError):
                if f:
                    f.close()

    def _call(self, req: dict, timeout: float | None) -> Any:
        proc = self._proc
        assert proc is not None
        assert proc.stdin is not None
        assert proc.stdout is not None
        try:
            worker.send(proc.stdin, req)
            if timeout is not None:
                # One reply per request, so the buffered reader never holds bytes select() cannot see.
                ready, _, _ = select.select([proc.stdout], [], [], timeout)
                if not ready:
                    self.kill()
                    raise EngineTimeout(f"{self.name}: no reply within {timeout:.0f}s")
            reply = worker.recv(proc.stdout)
        except (EOFError, BrokenPipeError, OSError) as e:
            code = proc.poll()
            self.kill()
            raise EngineCrashed(f"{self.name} worker died ({e!r}, exit {code})") from e
        if "error" in reply:
            raise EngineError(reply["error"])
        return reply["ok"]

    def _request(self, req: dict) -> Any:
        if self._proc is None:
            self._start()
        return self._call(req, self.timeout_s)

    # -- Engine -------------------------------------------------------------
    def setup(self, dataset: Dataset) -> None:
        self._dataset = dataset
        if self._proc is None:
            self._start()

    def version(self) -> str:
        return self.info.get("version", "?")

    def config(self) -> dict:
        return self.info.get("config", {})

    def build_info(self) -> dict[str, str]:
        return self.info.get("build", {})

    def run(self, query: Query) -> list[tuple]:
        return self._request({"op": "run", "query": query})

    def timed_run(self, query: Query) -> tuple[int, float]:
        """Runs ``query`` in the worker; the worker tree's peak RSS lands in ``last_peak_rss_bytes``."""
        self.last_peak_rss_bytes = None
        if self._proc is None:
            self._start()  # the window must cover the live worker, not a pid that does not exist yet
        self.begin_measure()
        try:
            return self._request({"op": "time", "query": query})
        finally:
            self.last_peak_rss_bytes = self.end_measure().peak_rss_bytes

    def profile(self, query: Query) -> dict:
        return self._request({"op": "profile", "query": query})

    def begin_measure(self) -> None:
        self.end_measure()
        if self._proc is not None:
            self._tree = TreeProc(self._proc.pid)
            self._tree.start()

    def end_measure(self) -> Measure:
        tree, self._tree = self._tree, None
        return Measure(peak_rss_bytes=tree.stop() if tree is not None else None)

    def cancel(self) -> None:
        self.kill()

    def teardown(self) -> None:
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            with contextlib.suppress(OSError):
                worker.send(proc.stdin, {"op": "quit"})
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=20)
        self.kill()
