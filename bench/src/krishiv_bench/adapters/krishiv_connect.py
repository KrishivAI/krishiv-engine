# SPDX-License-Identifier: Apache-2.0
"""``krishiv-connect``: TPC-H / TPC-DS / ClickBench through ``krishiv server`` over Spark Connect (P3-PR-53a).

The harness starts the ``krishiv`` binary as ``krishiv server --port 0 --connect-port 0`` on loopback, reads the
bound ``sc://`` address from its stdout, and drives it with the pinned pyspark client exactly as ``spark-connect``
does (Arrow results, ``interruptAll`` to cancel). The server runs on the host, so Parquet paths are not rewritten.
Peak RSS, CPU seconds and I/O are those of the server process tree (``TreeProc``), the same window the embedded
``krishiv`` engine reports, so the two differ by the Connect round trip and Arrow encoding only.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .base import Dataset, Measure
from .proc_mem import TreeProc, tree_measure
from .spark_connect import SparkConnectEngine

_ADDR = re.compile(r"Spark Connect on (sc://\S+)")


def server_binary(explicit: Path | str | None = None) -> Path:
    """An explicit path wins over ``$KRISHIV_BENCH_SERVER``, then ``<target>/dist/krishiv``."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("KRISHIV_BENCH_SERVER")
    if env:
        return Path(env)
    target = Path(os.environ.get("CARGO_TARGET_DIR", Path(__file__).resolve().parents[4] / "target"))
    return target / "dist" / "krishiv"


def parse_address(line: str) -> str | None:
    """The ``sc://host:port`` in the server's startup line, or None for any other line."""
    m = _ADDR.search(line)
    return m.group(1) if m else None


class KrishivConnectEngine(SparkConnectEngine):
    name = "krishiv-connect"

    def __init__(
        self,
        binary: Path | str | None = None,
        conf: list[tuple[str, str]] | None = None,
        startup_timeout_s: float = 60.0,
    ):
        super().__init__(remote="", container=None, container_data_root=None)
        self.binary = server_binary(binary)
        self.conf = list(conf or [])
        self.startup_timeout_s = startup_timeout_s
        self._proc: subprocess.Popen[str] | None = None
        self._tree: TreeProc | None = None
        self._log = Path(tempfile.mkdtemp(prefix="krishiv-connect-")) / "server.log"

    def command(self) -> list[str]:
        cmd = [str(self.binary), "server", "--port", "0", "--connect-port", "0"]
        for k, v in self.conf:
            cmd += ["--conf", f"{k}={v}"]
        return cmd

    def _start_server(self) -> None:
        if self._proc is not None:
            return
        if not self.binary.exists():
            raise FileNotFoundError(f"krishiv binary not found: {self.binary} (build it or set KRISHIV_BENCH_SERVER)")
        out = self._log.open("w")
        self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=out, text=True)
        assert self._proc.stdout is not None
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                break
            if (addr := parse_address(line)) is not None:
                self.remote = addr
                return
        self.teardown()
        raise RuntimeError(f"krishiv server did not report a Spark Connect address; see {self._log}")

    def version(self) -> str:
        return f"krishiv server ({self.binary.name})"

    def config(self) -> dict:
        return {"server": "krishiv server, loopback, Spark Connect", "binary": str(self.binary), "conf": self.conf}

    def setup(self, dataset: Dataset) -> None:
        self._start_server()
        super().setup(dataset)

    def begin_measure(self) -> None:
        self.end_measure()
        if self._proc is not None:
            self._tree = TreeProc(self._proc.pid)
            self._tree.start()

    def end_measure(self) -> Measure:
        tree, self._tree = self._tree, None
        return tree_measure(tree)

    def teardown(self) -> None:
        super().teardown()
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=15)
            if proc.poll() is None:
                proc.kill()
