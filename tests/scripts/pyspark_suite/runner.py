# SPDX-License-Identifier: Apache-2.0
"""Run the suite file by file against a fresh `krishiv server` per file (design §2)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from .junit import Record, parse

TEST_TIMEOUT_S = 120  # per test, recalibrated at the first measured run
FILE_TIMEOUT_S = 30 * 60


def _wait_port(port: int, deadline_s: float = 30.0) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


@contextmanager
def server(cmd: list[str], port: int) -> Generator[None]:
    """Start the server command, wait for its port, stop it (and wait) on exit."""
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not _wait_port(port):
            raise RuntimeError(f"server did not open port {port}: {cmd}")
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_file(version: str, checkout: Path, rel: str, out_dir: Path, server_cmd: list[str], port: int) -> list[Record]:
    """One file, one server. A crash or timeout yields a `timeout`/`error` record, never an abort."""
    tests_dir = checkout / "python" / "pyspark" / "sql" / "tests" / "connect"
    xml = out_dir / (rel.replace("/", "__") + ".xml")
    env = {k: v for k, v in os.environ.items() if k != "SPARK_HOME"}
    env["PYTHONPATH"] = str(checkout / "python")
    env["SPARK_CONNECT_TESTING_REMOTE"] = f"sc://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tests_dir / rel),
        f"--junitxml={xml}",
        "-p",
        "no:randomly",
        "-p",
        "no:cacheprovider",
        "-q",
        f"--timeout={TEST_TIMEOUT_S}",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with server(server_cmd, port):
        try:
            subprocess.run(cmd, env=env, timeout=FILE_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            return [Record(version, rel, "<file>", "<file>", "timeout", FILE_TIMEOUT_S * 1000)]
    return parse(xml, version, rel) if xml.exists() else [Record(version, rel, "<file>", "<file>", "error", 0)]
