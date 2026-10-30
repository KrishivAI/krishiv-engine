# SPDX-License-Identifier: Apache-2.0
"""A throw-away S3-compatible server (MinIO in ``docker run --rm``) that serves a remote-filter mirror read-only.

The container is named ``krishiv-bench-rf-<pid>`` and is always removed on exit, error or interrupt; the long-lived
``bench-minio`` / ``krishiv-bench-minio`` containers are never touched. Engines read ``s3://fw/<file>`` through the
endpoint and credentials in the environment variables of :meth:`S3Server.env`.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

# Same image as bench/baselines/docker-compose.yml (RELEASE.2025-09-07).
IMAGE = "minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
BUCKET = "fw"
ACCESS_KEY = "krishiv"
SECRET_KEY = "krishiv-bench-secret"
ENV_ENDPOINT, ENV_ACCESS, ENV_SECRET = "KRISHIV_BENCH_S3_ENDPOINT", "KRISHIV_BENCH_S3_KEY", "KRISHIV_BENCH_S3_SECRET"


class S3Server:
    """Context manager around one container; ``_run`` and ``_ready`` are injectable for tests."""

    def __init__(self, table_dir: Path, _run=subprocess.run, _ready=None, ready_timeout_s: float = 60):
        self.table_dir = Path(table_dir)
        self.name = f"krishiv-bench-rf-{os.getpid()}"
        self.endpoint = ""
        self._run = _run
        self._ready = _ready or self._http_ready
        self._ready_timeout_s = ready_timeout_s
        self._started = False

    def _docker(self, *args: str) -> str:
        try:
            done = self._run(["docker", *args], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"docker {args[0]} failed ({e.returncode}): {(e.stderr or '').strip()[:300]}") from e
        return (done.stdout or "").strip()

    def _http_ready(self, endpoint: str) -> bool:
        try:
            with urllib.request.urlopen(f"http://{endpoint}/minio/health/ready", timeout=2) as r:
                return r.status == 200
        except OSError:
            return False

    def start(self) -> S3Server:
        self._started = True  # set first: a failure below still removes whatever was created
        self._docker(
            "run", "--rm", "-d", "--name", self.name,
            "-p", "127.0.0.1::9000",
            "-v", f"{self.table_dir.resolve()}:/data/{BUCKET}:ro",
            "-e", f"MINIO_ROOT_USER={ACCESS_KEY}", "-e", f"MINIO_ROOT_PASSWORD={SECRET_KEY}",
            IMAGE, "server", "/data",
        )  # fmt: skip
        mapping = self._docker("port", self.name, "9000/tcp").splitlines()[0]
        self.endpoint = mapping.replace("0.0.0.0", "127.0.0.1")
        deadline = time.monotonic() + self._ready_timeout_s
        while not self._ready(self.endpoint):
            if time.monotonic() > deadline:
                raise RuntimeError(f"the S3 container {self.name} was not ready after {self._ready_timeout_s:.0f}s")
            time.sleep(0.5)
        return self

    def stop(self) -> None:
        if self._started:
            self._started = False
            # Best effort and idempotent: the container may already be gone (--rm).
            self._run(["docker", "rm", "-f", self.name], capture_output=True, text=True, check=False)

    def __enter__(self) -> S3Server:
        # Interrupts and exits remove the container too; the previous SIGTERM handler is restored on exit.
        self._atexit = atexit.register(self.stop)
        self._old_term = signal.signal(signal.SIGTERM, self._on_term)
        try:
            return self.start()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _on_term(self, signum, frame) -> None:
        self.stop()
        raise SystemExit(128 + signum)

    def __exit__(self, *exc) -> None:
        self.stop()
        atexit.unregister(self.stop)
        signal.signal(signal.SIGTERM, self._old_term)

    def urls(self) -> list[str]:
        return [f"s3://{BUCKET}/{p.name}" for p in sorted(self.table_dir.glob("*.parquet"))]

    def input_arg(self) -> str:
        """The JSON source list the children take."""
        return json.dumps(self.urls())

    def env(self) -> dict[str, str]:
        return {ENV_ENDPOINT: self.endpoint, ENV_ACCESS: ACCESS_KEY, ENV_SECRET: SECRET_KEY}


__all__ = ["BUCKET", "IMAGE", "S3Server"]
