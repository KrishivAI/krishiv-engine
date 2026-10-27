"""The ``remote-filter`` suite: a very selective scan of remote Parquet under a small memory scope.

One query (``remote_filter_queries.QUERY``) over the FineWeb-Edu sample mirror, run per engine in a fresh
child process (``remote_filter_child``) inside a ``capped`` scope. Scopes: the ``--cap`` scope plus an
unconstrained one (defaults ``4g`` and ``none``).
Correctness comes first, per engine: identical row count, identical ``SUM(token_count)`` and the identical
set of ``id`` values against the DuckDB reference run; an engine whose result differs is never timed.

Input modes: ``file`` (a plain local path: the fetched mirror or the synthetic stand-in, used for the
correctness run) and ``s3`` (the same files served by a throw-away MinIO container started with
``docker run --rm`` on a random port with random credentials: never an existing or shared container, never
a fixed container name; secrets travel in an ``--env-file`` and a cred file, never on a command line and
never in a recorded manifest). A mode for plain ``https://`` URLs is a later unit: ``HTTPS_MODE`` names it
and ``https_input`` refuses it loudly; there is no code for it.

Bytes fetched are counted by a counting proxy in front of the server (``CountingProxy``): engine-independent,
per run. Local reads traverse no object store, so file mode records ``null`` and the report says so instead
of guessing.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import http.client
import json
import os
import random
import secrets
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import remote_filter_data as data_mod
from .abba import bench_lock, bootstrap_ci, loadavg
from .capped import GIB, classify, read_result, run_scoped
from .paths import data_root, results_dir
from .remote_filter_queries import QUERIES
from .runner import git_info, machine_info
from .sizes import parse_size

ENGINES = ["krishiv", "duckdb", "polars", "spark"]
REFERENCE = "duckdb"
MODES = ("file", "s3")
HTTPS_MODE = "https"  # extension point: plain https:// input is a later unit (see https_input)
QUERY = QUERIES[0]
MINIO_IMAGE = "minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
S3_BUCKET = "remote-filter"
S3_REGION = "us-east-1"
THREADS = 8
REPS = 5
TIMEOUT_S = 1800.0


def https_input(url: str) -> str:
    """Placeholder for the plain-``https://`` input mode (a later unit): always refuses."""
    raise NotImplementedError(
        f"https:// input ({url}) is a later unit: no fetching, signing or counting code exists for it"
    )


# ---------------------------------------------------------------- S3 throw-away server
def random_port() -> int:
    return random.SystemRandom().randint(20000, 40000)


def new_s3_credentials() -> tuple[str, str]:
    """A random access key and secret for one throw-away server (never reused, never recorded)."""
    return ("rf-" + secrets.token_hex(4), secrets.token_hex(16))


def write_env_file(path: Path, user: str, password: str) -> None:
    """Writes the container's ``--env-file`` (owner-only); the secrets never appear on a command line."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"MINIO_ROOT_USER={user}\nMINIO_ROOT_PASSWORD={password}\n")


def s3_container_command(port: int, storage: Path, env_file: Path, image: str = MINIO_IMAGE) -> list[str]:
    """``docker run`` argv for the throw-away server: ``--rm``, loopback port, env file, no ``--name``.

    No fixed container name (docker assigns a random one) and no secret on the command line: credentials
    travel in ``env_file`` only. ``storage`` is a fresh writable directory (the server writes its metadata
    and the seeded objects there); the mirror is uploaded into it with ``seed_bucket``.
    """
    return [
        "docker",
        "run",
        "--rm",
        "-p",
        f"127.0.0.1:{port}:9000",
        "--env-file",
        str(env_file),
        "-v",
        f"{storage}:/data",
        image,
        "server",
        "/data",
    ]


# Minimal SigV4 S3 client (stdlib only, for seeding the throw-away server).
def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def s3_authorization(
    user: str,
    password: str,
    method: str,
    path: str,
    payload: bytes,
    host: str,
    amz_date: str,
    region: str = S3_REGION,
    payload_hash: str | None = None,
) -> dict[str, str]:
    """SigV4 headers for one request (path-style ``/bucket/key``). Only the signature leaves this function."""
    payload_hash = payload_hash or hashlib.sha256(payload).hexdigest()
    headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    canonical = (
        f"{method}\n{path}\n\n"
        + "".join(f"{k}:{headers[k]}\n" for k in sorted(headers))
        + "\n"
        + ";".join(sorted(headers))
        + f"\n{payload_hash}"
    )
    scope = f"{amz_date[:8]}/{region}/s3/aws4_request"
    to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    k = _hmac(("AWS4" + password).encode(), amz_date[:8])
    for part in (region, "s3", "aws4_request"):
        k = _hmac(k, part)
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    signed = ";".join(sorted(headers))
    credential = f"AWS4-HMAC-SHA256 Credential={user}/{scope}"
    return {**headers, "authorization": f"{credential}, SignedHeaders={signed}, Signature={signature}"}


def s3_request(
    endpoint: str,
    bucket: str,
    key: str,
    user: str,
    password: str,
    method: str = "GET",
    body: bytes | None = None,
) -> tuple[int, bytes]:
    """One S3 request against the throw-away server. Failures name method/path/status, never the secret."""
    parts = urllib.parse.urlsplit(endpoint)
    path = f"/{bucket}" + (f"/{key}" if key else "")
    amz_date = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    headers = s3_authorization(user, password, method, path, body or b"", parts.netloc, amz_date)
    conn = http.client.HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=60)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    except OSError as e:
        raise OSError(f"S3 {method} {path}: {type(e).__name__}: {e}") from None
    finally:
        conn.close()


def s3_make_bucket(endpoint: str, bucket: str, user: str, password: str) -> None:
    status, _ = s3_request(endpoint, bucket, "", user, password, method="PUT", body=b"")
    if status not in (200, 409):
        raise OSError(f"S3 PUT /{bucket}: unexpected status {status}")


def s3_put_file(endpoint: str, bucket: str, key: str, path: Path, user: str, password: str) -> None:
    """Streams ``path`` to the server (shards are gigabytes; never one ``read()``)."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    parts = urllib.parse.urlsplit(endpoint)
    url_path = f"/{bucket}/{key}"
    amz_date = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    headers = s3_authorization(user, password, "PUT", url_path, b"", parts.netloc, amz_date, payload_hash=h.hexdigest())
    conn = http.client.HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=600)
    try:
        conn.putrequest("PUT", url_path)
        for k, v in headers.items():
            conn.putheader(k, v)
        conn.putheader("Content-Length", str(size))
        conn.endheaders()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 23):
                conn.send(chunk)
        resp = conn.getresponse()
        resp.read()
        status = resp.status
    except OSError as e:
        raise OSError(f"S3 PUT {url_path}: {type(e).__name__}: {e}") from None
    finally:
        conn.close()
    if status != 200:
        raise OSError(f"S3 PUT /{bucket}/{key}: unexpected status {status}")


def s3_get_bytes(endpoint: str, bucket: str, key: str, user: str, password: str) -> bytes:
    status, body = s3_request(endpoint, bucket, key, user, password, method="GET")
    if status != 200:
        raise OSError(f"S3 GET /{bucket}/{key}: unexpected status {status}")
    return body


def s3_get_sha256(endpoint: str, bucket: str, key: str, user: str, password: str) -> str:
    """Streams one object back and returns its SHA-256 (shards are gigabytes; never one ``read()``)."""
    parts = urllib.parse.urlsplit(endpoint)
    url_path = f"/{bucket}/{key}"
    amz_date = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    headers = s3_authorization(user, password, "GET", url_path, b"", parts.netloc, amz_date)
    h = hashlib.sha256()
    conn = http.client.HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=600)
    try:
        conn.request("GET", url_path, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        if status == 200:
            while chunk := resp.read(1 << 23):
                h.update(chunk)
    except OSError as e:
        raise OSError(f"S3 GET {url_path}: {type(e).__name__}: {e}") from None
    finally:
        conn.close()
    if status != 200:
        raise OSError(f"S3 GET /{bucket}/{key}: unexpected status {status}")
    return h.hexdigest()


class S3Server:
    """A throw-away MinIO on a random port with random credentials. Never an existing/shared container."""

    def __init__(self, image: str = MINIO_IMAGE):
        self.image = image
        self.port = random_port()
        self.user, self.password = new_s3_credentials()
        self.root = Path(f"/tmp/opencode/remote-filter-s3-{os.getpid()}-{self.port}")
        self.proc: subprocess.Popen | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> S3Server:
        self.root.mkdir(parents=True, exist_ok=True)
        storage = self.root / "storage"
        storage.mkdir(exist_ok=True)
        env_file = self.root / "minio.env"
        write_env_file(env_file, self.user, self.password)
        cmd = s3_container_command(self.port, storage, env_file, self.image)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise OSError(f"throw-away S3 server exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.endpoint}/minio/health/ready", timeout=2) as r:
                    if r.status == 200:
                        return self
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise OSError(f"throw-away S3 server on port {self.port} never became ready")

    def seed(self, docs_dir: Path) -> list[str]:
        """Uploads every shard of ``docs_dir`` to ``s3://BUCKET/docs/``; returns the keys."""
        s3_make_bucket(self.endpoint, S3_BUCKET, self.user, self.password)
        keys = []
        for f in sorted(docs_dir.glob("*.parquet")):
            key = f"docs/{f.name}"
            s3_put_file(self.endpoint, S3_BUCKET, key, f, self.user, self.password)
            keys.append(key)
        if not keys:
            raise FileNotFoundError(f"nothing to seed the S3 server with under {docs_dir}")
        return keys

    def verify_seed(self, docs_dir: Path) -> None:
        """Reads one seeded shard back and compares digests: the server serves the mirror, nothing else."""
        shards = sorted(docs_dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"nothing to seed the S3 server with under {docs_dir}")
        key = f"docs/{shards[0].name}"
        got = s3_get_sha256(self.endpoint, S3_BUCKET, key, self.user, self.password)
        want = data_mod.sha256_file(shards[0])
        if got != want:
            raise OSError(f"seed verification failed for /{S3_BUCKET}/{key}: digest {got} != {want}")

    def stop(self) -> None:
        if self.proc is not None:
            proc, self.proc = self.proc, None
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------- counting proxy (engine-independent bytes-fetched)
_HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "upgrade"}


class CountingProxy:
    """Engines point at the proxy; it forwards to the server (the original ``Host`` header is kept, so
    SigV4 signatures stay valid; framing is normalized to Content-Length) and counts response bodies.
    ``response_bytes`` is the run's ``fetched_bytes``.
    """

    def __init__(self, target_host: str, target_port: int):
        self.target_host, self.target_port = target_host, target_port
        self._bytes = 0
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CountingProxy/1"

            def _proxy(self) -> None:
                length = self.headers.get("Content-Length")
                body = self.rfile.read(int(length)) if length else b""
                if self.headers.get("Expect", "").lower() == "100-continue":
                    self.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
                conn = http.client.HTTPConnection(outer.target_host, outer.target_port, timeout=300)
                try:
                    conn.request(self.command, self.path, body=body or None, headers=headers)
                    resp = conn.getresponse()
                    data = resp.read()
                finally:
                    conn.close()
                with outer._lock:
                    outer._bytes += len(data)
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in _HOP | {"content-length"}:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def __getattr__(self, name: str):  # any HTTP method proxies
                if name.startswith("do_"):
                    return self._proxy
                raise AttributeError(name)

            def log_message(self, format: str, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def response_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)


# ---------------------------------------------------------------- correctness
def _glob(out: Path) -> str:
    return str(out / "**" / "*.parquet").replace("'", "''")


def checksum(out: Path) -> dict:
    """The cross-engine checksum of a result: row count and exact ``SUM(token_count)`` (DuckDB reads it)."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    cur = con.execute(
        f"SELECT COUNT(*) AS n, CAST(SUM(token_count) AS BIGINT) AS sum_token FROM read_parquet('{_glob(out)}')"
    )
    cols = [d[0] for d in cur.description or []]
    row = cur.fetchone()
    return dict(zip(cols, row if row else [None] * len(cols), strict=True))


def compare_results(ref_out: Path, got_out: Path) -> str | None:
    """None when ``got`` matches the reference: same row count, same sum, same set of ``id`` values."""
    import duckdb

    ref, got = checksum(ref_out), checksum(got_out)
    if ref["n"] != got["n"]:
        return f"n: reference {ref['n']!r}, got {got['n']!r}"
    if ref["sum_token"] != got["sum_token"]:
        return f"sum_token: reference {ref['sum_token']!r}, got {got['sum_token']!r}"
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    for a, b, side in ((ref_out, got_out, "missing"), (got_out, ref_out, "extra")):
        row = con.execute(
            f"SELECT COUNT(*) FROM (SELECT id FROM read_parquet('{_glob(a)}') "
            f"EXCEPT SELECT id FROM read_parquet('{_glob(b)}'))"
        ).fetchone()
        assert row is not None
        if row[0]:
            return f"id set differs: {row[0]} {side} rows"
    return None


# ---------------------------------------------------------------- one run
def _write_cred_file(path: Path, user: str, password: str) -> None:
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"AWS_ACCESS_KEY_ID={user}\nAWS_SECRET_ACCESS_KEY={password}\n")


def run_one(
    engine: str,
    data: Path,
    engine_memory: str,
    cap_bytes: int | None,
    timeout_s: float | None,
    work: Path,
    *,
    mode: str = "file",
    s3: dict | None = None,
) -> dict:
    """One fresh-process run under the (possibly unconstrained) scope.

    ``data`` is the local mirror; in ``s3`` mode it seeds the server and the child reads
    ``s3://BUCKET/docs`` through the counting proxy. Returns the run record (``fetched_bytes`` is the
    proxy count in ``s3`` mode, ``None`` for local reads, which traverse no object store).
    """
    if mode == HTTPS_MODE:
        https_input(f"{HTTPS_MODE}://…")
    if mode not in MODES:
        raise ValueError(f"unknown remote-filter mode {mode!r}; available: {', '.join(MODES)}")
    out, scratch = work / "out", work / "scratch"
    proxy: CountingProxy | None = None
    cred_file = work / "s3creds.env"
    work.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "s3":
            if s3 is None:
                raise ValueError("s3 mode needs the server's connection info")
            _write_cred_file(cred_file, s3["user"], s3["password"])
            proxy = CountingProxy(s3["host"], s3["port"])
            child_input = f"s3://{s3['bucket']}/docs"
            endpoint = proxy.url
        else:
            child_input = str(data / data_mod.DOCS_DIR if (data / data_mod.DOCS_DIR).is_dir() else data)
            endpoint = None
        argv = [
            sys.executable,
            "-m",
            "krishiv_bench.remote_filter_child",
            engine,
            child_input,
            QUERY,
            engine_memory,
            str(out),
            str(scratch),
        ]
        if endpoint is not None:
            argv += ["--s3-endpoint", endpoint, "--cred-file", str(cred_file)]
        before = proxy.response_bytes if proxy else 0
        obs, stdout_text, err_tail = run_scoped(
            argv,
            env_extra={"POLARS_MAX_THREADS": "8"},
            cap_bytes=cap_bytes,
            timeout_s=timeout_s,
            work=work,
            scratch=scratch,
        )
        fetched = (proxy.response_bytes - before) if proxy else None
        result = read_result(stdout_text)
        status = classify(obs, cap_bytes, result is not None)
        rec = {
            "engine": engine,
            "query": QUERY,
            "mode": mode,
            "status": status,
            "wall_s": round(obs.wall_s, 3),
            "peak_rss_bytes": obs.rss_peak_bytes,
            "peak_cgroup_bytes": obs.peak_bytes,
            "spill_bytes": obs.spill_peak_bytes,
            "fetched_bytes": fetched,
            "returncode": obs.returncode,
            "oom_kills": obs.oom_kills,
        }
        if status == "finished" and result is not None:
            rec |= {k: result[k] for k in ("version", "setup_s", "query_s", "rows_out")}
        else:
            rec["error"] = err_tail[:500]
        return rec
    finally:
        if proxy is not None:
            proxy.close()
        with contextlib.suppress(OSError):
            cred_file.unlink()


# ---------------------------------------------------------------- suite
def abba_order(order: list[str], rep: int) -> list[str]:
    """ABBA-generalized engine order of repetition ``rep``: forward, reverse, reverse, forward, ..."""
    return order if rep % 4 in (0, 3) else list(reversed(order))


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def median_ci(xs: list[float], seed: int = 20261027, iters: int = 2000) -> list[float] | None:
    """Bootstrap 95% CI of the median (same resampling spirit as ``abba.bootstrap_ci``)."""
    if not xs:
        return None
    rng = random.Random(seed)
    ms = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(iters))
    return [ms[int(0.025 * iters)], ms[int(0.975 * iters) - 1]]


def summarize(runs: list[dict], ref_walls: list[float] | None) -> dict:
    """Timed runs of one (engine, scope): counts, medians, wall CI, ratio vs the reference, fetched bytes."""
    timed = [r for r in runs if r["phase"] == "timed"]
    done = [r for r in timed if r["status"] == "finished"]
    walls = [r["wall_s"] for r in done]
    fetched = [r["fetched_bytes"] for r in done if r.get("fetched_bytes") is not None]
    s: dict = {
        "n": len(timed),
        "finished": len(done),
        "killed": sum(r["status"] == "killed" for r in timed),
        "median_wall_s": median(walls),
        "wall_ci95": median_ci(walls),
        "median_peak_rss_bytes": median([r["peak_rss_bytes"] for r in done]),
        "max_peak_rss_bytes": max((r["peak_rss_bytes"] for r in timed), default=None),
        "median_spill_bytes": median([r["spill_bytes"] for r in done if r.get("spill_bytes") is not None]),
        "median_fetched_bytes": median(fetched) if fetched else None,
        "ratio_vs_reference": None,
        "ratio_ci95": None,
    }
    if ref_walls and walls and s["median_wall_s"] and median(ref_walls):
        lo, hi = bootstrap_ci(walls, ref_walls)
        s["ratio_vs_reference"] = s["median_wall_s"] / (median(ref_walls) or 1)
        s["ratio_ci95"] = [lo, hi]
    return s


def run_query(scope: str, order: list[str], reps: int, go, check) -> dict:
    """Validation pass (every engine, full checksum vs the DuckDB reference in the same mode), then ``reps``
    timed repetitions in ABBA engine order. A mismatching engine is never timed."""
    ref_out: Path | None = None
    verdict: dict[str, str] = {}
    runs: dict[str, list[dict]] = {e: [] for e in order}

    def verify(engine: str, rec: dict, out: Path) -> None:
        nonlocal ref_out
        try:
            if engine == REFERENCE:
                ref_out = out
                got = None
            elif ref_out is None:
                rec["verdict"] = verdict[engine] = "unverified: no reference result"
                return
            else:
                got = check(ref_out, out)
        except Exception as e:
            rec["verdict"] = verdict[engine] = f"unverified: {type(e).__name__}: {str(e)[:200]}"
            return
        finally:
            if engine != REFERENCE:
                shutil.rmtree(out.parent, ignore_errors=True)
        if engine == REFERENCE:
            rec["verdict"] = verdict[engine] = "reference"
        elif got is None:
            rec["verdict"] = verdict[engine] = "ok"
        else:
            rec["verdict"] = verdict[engine] = f"mismatch: {got}" if got else "ok"

    for e in order:
        rec, out = go(e, True)
        rec["phase"] = "validation"
        runs[e].append(rec)
        if rec["status"] == "finished":
            verify(e, rec, out)
        else:
            verdict[e] = "unverified: validation run " + rec["status"]
    for rep in range(reps):
        for e in abba_order(order, rep):
            if verdict[e].startswith("mismatch"):
                continue  # a wrong result is never timed
            pending = not (verdict[e] in ("ok", "reference") or verdict[e].startswith("mismatch"))
            rec, out = go(e, pending)
            rec["phase"] = "timed"
            rec["rep"] = rep
            rec["scope"] = scope
            runs[e].append(rec)
            if rec["status"] != "finished":
                continue
            if pending:
                verify(e, rec, out)
            else:
                rec["verdict"] = verdict[e]
            if verdict[e].startswith("mismatch"):
                rec["status"] = "mismatch"
    ref_walls = [r["wall_s"] for r in runs[REFERENCE] if r.get("phase") == "timed" and r["status"] == "finished"]
    return {
        "engines": {
            e: {
                "verdict": verdict[e],
                "runs": runs[e],
                "summary": summarize(runs[e], ref_walls if e != REFERENCE else None),
            }
            for e in order
        },
    }


def parse_caps(cap: str) -> list[tuple[str, int | None]]:
    """``--cap`` plus the unconstrained scope: ``4g`` -> [(``4g``, bytes), (``none``, None)]."""
    if cap.strip().lower() == "none":
        return [("none", None)]
    return [(cap.lower(), parse_size(cap)), ("none", None)]


def run_remote_filter(
    data: Path,
    mode: str = "file",
    cap: str = "4g",
    engines: list[str] | None = None,
    engine_memory: str = "3g",
    reps: int = REPS,
    timeout_s: float | None = TIMEOUT_S,
    run=run_one,
    check=compare_results,
    serve=S3Server,
) -> Path:
    """Runs the suite on ``data`` (a fetched mirror or the synthetic stand-in) and writes the result JSON."""
    if mode == HTTPS_MODE:
        https_input(f"{HTTPS_MODE}://…")
    if mode not in MODES:
        raise ValueError(
            f"unknown remote-filter mode {mode!r}; available: {', '.join(MODES)} (plus {HTTPS_MODE}, later)"
        )
    if reps < 1:
        raise ValueError(f"--reps must be at least 1 (got {reps})")
    if engines is not None and not engines:
        raise ValueError(f"engines must be a non-empty subset of {', '.join(ENGINES)} (got {engines})")
    engines = list(engines or ENGINES)
    unknown = [e for e in engines if e not in ENGINES]
    if unknown:
        raise ValueError(f"engines must be a non-empty subset of {', '.join(ENGINES)} (got {engines})")
    parse_size(engine_memory)
    digest = data_mod.load_digest(data)
    scopes = parse_caps(cap)
    order = [REFERENCE, *[e for e in engines if e != REFERENCE]]  # the reference runs first in validation
    record: dict = {
        "suite": "remote-filter",
        "kind": "remote-filter",
        "ts": time.strftime("%Y%m%dT%H%M%S"),
        "query": QUERY,
        "mode": mode,
        "cap": cap.lower(),
        "engine_memory": engine_memory,
        "threads": THREADS,
        "reps": reps,
        "reported_engines": engines,
        "machine": machine_info(),
        "git": git_info(),
        "dataset": {"path": str(data), **digest},
        "scopes": {},
    }
    server: S3Server | None = None
    if mode == "s3":
        docs = data / data_mod.DOCS_DIR
        server = serve().start()
        try:
            seed_dir = docs if docs.is_dir() else data
            keys = server.seed(seed_dir)
            server.verify_seed(seed_dir)
        except BaseException:
            server.stop()
            raise
        record["s3"] = {"endpoint_host": "127.0.0.1", "bucket": S3_BUCKET, "image": server.image, "keys": len(keys)}
    work_root = data_root() / "tmp" / f"remote-filter-{os.getpid()}"
    n = 0
    s3info = (
        {
            "host": "127.0.0.1",
            "port": server.port,
            "bucket": S3_BUCKET,
            "user": server.user,
            "password": server.password,
        }
        if server
        else None
    )

    def go(scope: str, cap_bytes: int | None, engine: str, keep: bool) -> tuple[dict, Path]:
        nonlocal n
        n += 1
        work = work_root / f"{n}-{scope}-{engine}"
        rec = run(engine, data, engine_memory, cap_bytes, timeout_s, work, mode=mode, s3=s3info)
        if not (keep and rec["status"] == "finished"):
            shutil.rmtree(work, ignore_errors=True)
        return rec, work / "out"

    try:
        with bench_lock() as waited:
            record["bench_lock_wait_s"] = waited
            record["loadavg_start"] = loadavg()
            try:
                for scope, cap_bytes in scopes:
                    record["scopes"][scope] = run_query(
                        scope,
                        order,
                        reps,
                        lambda e, keep, _s=scope, _c=cap_bytes: go(_s, _c, e, keep),
                        check,
                    )
            finally:
                shutil.rmtree(work_root, ignore_errors=True)
            record["loadavg_end"] = loadavg()
    finally:
        if server is not None:
            server.stop()
    versions = {}
    for per_scope in record["scopes"].values():
        for e, d in per_scope["engines"].items():
            for r in d["runs"]:
                if r.get("version"):
                    versions[e] = r["version"]
    record["engine_versions"] = versions
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{record['ts']}_remotefilter_{mode}_{'-'.join(s for s, _ in scopes)}_{engine_memory.lower()}.json"
    redacted = json.dumps(record, indent=2)
    if server is not None and (server.user in redacted or server.password in redacted):
        raise AssertionError("credentials leaked into the recorded result")
    f.write_text(redacted)
    print(f"result: {f}")
    return f


# ---------------------------------------------------------------- report
def latest_remote_filter(
    mode: str, directory: Path | None = None
) -> dict[tuple[str, str, str], tuple[str, dict, dict]]:
    """Newest result per (scope, engine, engine memory): ``(file name, run record, engine entry)``."""
    best: dict[tuple[str, str, str], tuple[str, dict, dict]] = {}
    for f in sorted((directory or results_dir()).glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        if r.get("kind") != "remote-filter" or r.get("mode") != mode:
            continue
        for scope, per_scope in r["scopes"].items():
            for e in r["reported_engines"]:
                if e in per_scope["engines"]:
                    best[(scope, e, r["engine_memory"])] = (f.name, r, per_scope["engines"][e])
    return best


def _gib(x: float | None) -> str:
    return "-" if x is None else f"{x / GIB:.2f}"


def _bytes(x: int | None) -> str:
    if x is None:
        return "-"
    return f"{x / GIB:.2f} GiB" if x >= GIB else f"{x / (1 << 20):.1f} MiB"


def render_remote_filter(mode: str, directory: Path | None = None) -> str:
    best = latest_remote_filter(mode, directory)
    if not best:
        raise SystemExit(f"no remote-filter results for mode {mode} under {directory or results_dir()}")
    first = next(iter(best.values()))[1]
    scopes = sorted({s for s, _, _ in best})
    L = [f"# remote-filter report: {QUERY} ({mode})", ""]
    L += [
        "One very selective filter over remote Parquet; a run killed above its cap counts as failed, never timed.",
        f"- Machine: {first['machine']['cpu_model']}, {first['machine']['cores_logical']} logical cores, "
        f"{first['machine']['ram_bytes'] / GIB:.0f} GiB RAM, {first['machine']['os']}",
        f"- Scopes: {', '.join(scopes)}; threads {first['threads']}; engine memory {first['engine_memory']}",
        f"- Dataset: `{first['dataset']['path']}`",
        "- Wall time includes interpreter and session start (and the scope's start). Peak memory is the highest "
        "sampled resident set of the whole process tree. Spill is the peak size of the engine's scratch directory "
        "(`-`: none or not measurable).",
        "- Fetched bytes are counted by the proxy in front of the throw-away S3 server (`-` in file mode: local "
        "reads traverse no object store, so there is no engine-independent count and none is guessed).",
        "- Medians are over finished runs; `peak` is `>cap (killed)` when no run finished. Wall time shows the "
        "bootstrap 95% CI of the median.",
        "",
        "| Engine | Scope | Engine memory | Median wall (s) | Finished | Peak memory (GiB) | Fetched "
        "| Spill (GiB) | Verdict | Result file |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (scope, e, mem), (fname, r, d) in sorted(best.items()):
        s = d["summary"]
        if s["median_wall_s"] is None:
            wall = "-"
        elif s["wall_ci95"]:
            wall = f"{s['median_wall_s']:.2f} ({s['wall_ci95'][0]:.2f}-{s['wall_ci95'][1]:.2f})"
        else:
            wall = f"{s['median_wall_s']:.2f}"
        if s["finished"]:
            peak = _gib(s["median_peak_rss_bytes"])
        else:
            cap_b = next((c for sc, c in parse_caps(r["cap"]) if sc == scope), None)
            peak = f">{cap_b / GIB:.1f} (killed)" if cap_b and s["killed"] else _gib(s["max_peak_rss_bytes"])
        verdict = "pass" if d["verdict"] in ("ok", "reference") and s["finished"] else d["verdict"]
        L.append(
            f"| {e} | {scope} | {mem} | {wall} | {s['finished']}/{s['n']} | {peak} | "
            f"{_bytes(s['median_fetched_bytes'])} | {_gib(s['median_spill_bytes'])} | {verdict} | `{fname}` |"
        )
    return "\n".join(L) + "\n"


def write_remote_filter_report(mode: str) -> Path:
    md = render_remote_filter(mode)
    out = results_dir() / f"report_remotefilter_{mode}.md"
    out.write_text(md)
    return out


__all__ = [
    "ENGINES",
    "HTTPS_MODE",
    "MINIO_IMAGE",
    "MODES",
    "QUERY",
    "REFERENCE",
    "REPS",
    "S3_BUCKET",
    "TIMEOUT_S",
    "CountingProxy",
    "S3Server",
    "abba_order",
    "checksum",
    "compare_results",
    "https_input",
    "latest_remote_filter",
    "median_ci",
    "new_s3_credentials",
    "parse_caps",
    "random_port",
    "render_remote_filter",
    "run_one",
    "run_query",
    "run_remote_filter",
    "s3_authorization",
    "s3_container_command",
    "s3_get_bytes",
    "s3_get_sha256",
    "write_env_file",
    "write_remote_filter_report",
]
