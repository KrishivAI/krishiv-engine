# SPDX-License-Identifier: Apache-2.0
"""Run the suite file by file against a fresh `krishiv server` per file (design §2)."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

from .junit import Record, parse

TEST_TIMEOUT_S = 120  # per test, recalibrated at the first measured run
FILE_TIMEOUT_S = 30 * 60


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_port(port: int, proc: subprocess.Popen, deadline_s: float = 30.0) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            return False
        if _port_open(port):
            return True
        time.sleep(0.2)
    return False


def check_pytest_timeout() -> None:
    """Without pytest-timeout every `--timeout` run errors out and shows as 'not run'; fail early."""
    if subprocess.run([sys.executable, "-c", "import pytest_timeout"], check=False, capture_output=True).returncode:
        raise RuntimeError("pytest-timeout is not installed in the version venv")


def stage_tests(checkout: Path) -> Path:
    """Copy the tag's test sources into the installed `pyspark` and return its `sql/tests/connect` dir.

    The installed wheel stays the only `pyspark` on the path: a sparse checkout on `PYTHONPATH` would shadow it
    with a partial package (no `pyspark.logger`, `pyspark.sql.connect`) and nothing would import.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import pathlib, pyspark; print(pathlib.Path(pyspark.__file__).parent)"],
        check=False,
        capture_output=True,
        text=True,
        cwd="/",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    if probe.returncode:
        raise RuntimeError(f"pyspark is not installed in the version venv: {probe.stderr.strip()}")
    installed = Path(probe.stdout.strip())
    src = checkout / "python" / "pyspark"
    for sub in ("sql/tests", "testing"):
        if not (src / sub).is_dir():
            raise RuntimeError(f"{src / sub} is missing from the Spark checkout")
        shutil.copytree(src / sub, installed / sub, dirs_exist_ok=True)
    return installed / "sql" / "tests" / "connect"


@contextmanager
def server(cmd: list[str], port: int) -> Generator[None]:
    """Start the server command, wait for its port, stop it (and wait) on exit."""
    if _port_open(port):
        raise RuntimeError(f"port {port} already has a listener; refusing to test a stale server")
    # Own process group, so a wrapper (`cargo run`, a script) cannot leave the real server behind.
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        if not _wait_port(port, proc):
            raise RuntimeError(f"server did not open port {port} (or exited early): {cmd}")
        yield
    finally:
        _killpg(proc, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            proc.wait(10)
        # The leader may have exited while a child ignored SIGTERM: always finish the group off.
        _killpg(proc, signal.SIGKILL)
        proc.wait()


def _killpg(proc: subprocess.Popen, sig: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, sig)


def run_file(version: str, tests_dir: Path, rel: str, out_dir: Path, server_cmd: list[str], port: int) -> list[Record]:
    """One file under `tests_dir` (the staged tests), one server; a crash or timeout is a record, not an abort."""
    xml = out_dir / (rel.replace("/", "__") + ".xml")
    env = {k: v for k, v in os.environ.items() if k != "SPARK_HOME"}
    env.pop("PYTHONPATH", None)  # the installed pyspark is the only one (see `stage_tests`)
    env["SPARK_CONNECT_TESTING_REMOTE"] = f"sc://127.0.0.1:{port}"
    # Remote-only mode, as with pyspark-client: no JVM session behind the tests (from p3/p31-pb01-nightly).
    env["SPARK_SKIP_CONNECT_COMPAT_TESTS"] = "1"
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
    xml.unlink(missing_ok=True)  # a crash that writes nothing must not pick up the last run's results
    with server(server_cmd, port):
        try:
            subprocess.run(cmd, env=env, timeout=FILE_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            return [Record(version, rel, "<file>", "<file>", "timeout", FILE_TIMEOUT_S * 1000)]
    return parse(xml, version, rel) if xml.exists() else [Record(version, rel, "<file>", "<file>", "error", 0)]
