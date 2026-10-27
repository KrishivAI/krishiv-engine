"""remote-filter suite: synthetic data, query spelling, caps, checksums, S3 scaffolding, orchestration."""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_memcap import FakeCgroup, FakeProc

from krishiv_bench import remote_filter, remote_filter_data, remote_filter_queries
from krishiv_bench.remote_filter import CountingProxy
from krishiv_bench.remote_filter import run_remote_filter as _run_real

TINY_ROWS = 3_000


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("remote-filter") / "synthetic"
    remote_filter_data.gen_synthetic(TINY_ROWS, out)
    return out


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    return tmp_path


# ------------------------------------------------------------------ synthetic data
def test_synthetic_schema_types_and_survivors(tiny):
    t = pq.read_table(tiny / "docs.parquet")
    assert t.num_rows == TINY_ROWS
    assert t.schema == remote_filter_data.SCHEMA
    assert [f.name for f in t.schema] == ["id", "text", "token_count", "int_score", "language_score"]
    rows = (
        duckdb.connect()
        .execute(f"SELECT COUNT(*) FROM read_parquet('{tiny}/docs.parquet') WHERE {remote_filter_queries.FILTER}")
        .fetchone()
    )
    assert rows is not None
    got = rows[0]
    assert got == remote_filter_data.SYNTHETIC_SURVIVORS


def test_synthetic_is_deterministic(tiny, tmp_path):
    again = tmp_path / "again"
    rec = remote_filter_data.gen_synthetic(TINY_ROWS, again)
    first = json.loads((tiny / "digest.json").read_text())
    assert rec["content_sha256"] == first["content_sha256"] == rec["content_sha256"]
    assert rec["survivors"] == remote_filter_data.SYNTHETIC_SURVIVORS
    assert remote_filter_data.load_digest(again) == rec


def test_synthetic_never_overwrites_and_rejects_tiny_counts(tiny, tmp_path):
    before = sorted(p.name for p in tiny.rglob("*"))
    with pytest.raises(FileExistsError, match="never regenerated in place"):
        remote_filter_data.gen_synthetic(TINY_ROWS, tiny)
    assert sorted(p.name for p in tiny.rglob("*")) == before
    with pytest.raises(ValueError, match="must exceed the survivor count"):
        remote_filter_data.gen_synthetic(remote_filter_data.SYNTHETIC_SURVIVORS, tmp_path / "x")


def test_mirror_helpers_and_missing_digest(workdir):
    assert remote_filter_data.mirror_dir() == workdir / "remote-filter" / "v1"
    assert remote_filter_data.local_name("sample/10BT/000_00000.parquet") == "000_00000.parquet"
    url = remote_filter_data.shard_url("sample/10BT/000_00000.parquet")
    assert url.startswith("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/")
    assert remote_filter_data.REVISION in url
    assert "://" not in url.split("/resolve/")[1].split("/")[0]  # nothing secret in the URL
    assert len(remote_filter_data.SHARD_FILES) == 14
    with pytest.raises(FileNotFoundError, match="krishiv-datasets remote-filter"):
        remote_filter_data.load_digest(workdir / "remote-filter" / "v1")


def test_fetch_from_prefetched_mirror_records_revision_and_digests(workdir, monkeypatch, tmp_path):
    monkeypatch.setattr(remote_filter_data, "SHARD_FILES", ["sample/10BT/a.parquet", "sample/10BT/b.parquet"])
    mirror = tmp_path / "prefetch"
    (mirror).mkdir()
    for name in ("a.parquet", "b.parquet"):
        pq.write_table(pa.table({"id": [name]}), mirror / name)
    out = remote_filter_data.fetch(mirror=mirror)
    assert out == workdir / "remote-filter" / "v1"
    digest = remote_filter_data.load_digest(out)
    assert digest["revision"] == remote_filter_data.REVISION
    assert digest["source"] == remote_filter_data.SOURCE_REPO
    assert digest["config"] == remote_filter_data.SOURCE_CONFIG
    assert [f["name"] for f in digest["files"]] == ["docs/a.parquet", "docs/b.parquet"]
    for f in digest["files"]:
        assert f["sha256"] == remote_filter_data.sha256_file(out / f["name"])
    manifest = workdir / "manifests" / f"{remote_filter_data.MANIFEST_NAME}.sha256"
    assert manifest.is_file()
    assert f"docs/{digest['files'][0]['name'].split('/')[-1]}" in manifest.read_text()
    assert remote_filter_data.fetch() == out  # an existing digest is reused, never regenerated
    (tmp_path / "incomplete").mkdir()
    pq.write_table(pa.table({"id": ["a"]}), tmp_path / "incomplete" / "a.parquet")
    with pytest.raises(FileNotFoundError, match="lacks"):
        remote_filter_data.fetch(dest=workdir / "other", mirror=tmp_path / "incomplete")
    bad = workdir / "remote-filter" / "v9"
    bad.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="never regenerated in place"):
        remote_filter_data.fetch(dest=bad)


# ------------------------------------------------------------------ queries
def test_query_spelling_and_s3_fragments():
    assert remote_filter_queries.QUERY == "filter_edu"
    sql = remote_filter_queries.sql_for("krishiv", "filter_edu")
    assert "int_score >= 4 AND token_count >= 4000 AND language_score >= 0.95" in sql
    assert remote_filter_queries.sql_for("duckdb", "filter_edu") == sql  # one spelling today
    with pytest.raises(ValueError, match="unknown remote-filter query"):
        remote_filter_queries.sql_for("duckdb", "q9")
    duck = remote_filter_queries.s3_sql("duckdb", "filter_edu", "s3://b/docs/*.parquet")
    assert duck.startswith("SELECT * FROM read_parquet('s3://b/docs/*.parquet')")
    assert "read_parquet('s3://b/docs" in duck
    assert "secret" not in duck.lower()
    krishiv = remote_filter_queries.s3_sql("krishiv", "filter_edu", "s3://b/docs/*.parquet")
    assert krishiv.startswith("SELECT * FROM parquet.`s3://b/docs/*.parquet`")
    with pytest.raises(ValueError, match="native reader"):
        remote_filter_queries.s3_from("polars", "s3://b/docs/*.parquet")
    with pytest.raises(NotImplementedError, match="later unit"):
        remote_filter.https_input("https://example.com/x.parquet")


def _engines_available() -> dict[str, str]:
    from krishiv_bench.adapters.krishiv import runner_binary

    out = {"duckdb": "", "polars": ""}
    if runner_binary().exists():
        out["krishiv"] = ""
    else:
        out["krishiv"] = "no dist runner built (KRISHIV_BENCH_RUNNER missing)"
    if shutil.which("java") is None:
        out["spark"] = "no Java on PATH"
    return out


def test_query_spellings_agree_on_the_synthetic_data(tiny, tmp_path):
    from krishiv_bench import remote_filter as rf
    from krishiv_bench.remote_filter_child import main

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    assert main(["duckdb", str(tiny), "filter_edu", "1g", str(ref_dir / "out"), str(ref_dir / "s")]) == 0
    for engine, reason in _engines_available().items():
        if reason:
            pytest.skip(reason)
        work = tmp_path / engine
        assert main([engine, str(tiny), "filter_edu", "1g", str(work / "out"), str(work / "s")]) == 0
        assert rf.compare_results(ref_dir / "out", work / "out") is None, engine
        if engine in ("duckdb", "polars"):
            target = work / "out" / "result.parquet"
            assert target.is_file()  # one file, not a directory of parts
            assert len(list((work / "out").rglob("*.parquet"))) == 1
            assert pq.ParquetFile(target).metadata.row_group(0).num_columns == 5
            assert "ZSTD" in {pq.ParquetFile(target).metadata.row_group(0).column(i).compression for i in range(5)}


def test_child_results_match_a_handwritten_oracle(tiny, tmp_path):
    from krishiv_bench.remote_filter_child import main

    work = tmp_path / "w"
    assert main(["duckdb", str(tiny), "filter_edu", "1g", str(work / "out"), str(work / "s")]) == 0
    oracle_row = (
        duckdb.connect()
        .execute(
            f"SELECT COUNT(*), SUM(token_count) FROM read_parquet('{tiny}/docs.parquet') "
            f"WHERE {remote_filter_queries.FILTER}"
        )
        .fetchone()
    )
    assert oracle_row is not None
    check = remote_filter.checksum(work / "out")
    assert (check["n"], check["sum_token"]) == (oracle_row[0], oracle_row[1])
    assert check["n"] == remote_filter_data.SYNTHETIC_SURVIVORS


def test_child_rejects_unknown_engine_and_missing_endpoint(tiny, tmp_path):
    from krishiv_bench.remote_filter_child import apply_cred_file, main

    with pytest.raises(SystemExit, match="unknown remote-filter engine 'flink'"):
        main(["flink", str(tiny), "filter_edu", "1g", str(tmp_path / "o"), str(tmp_path / "s")])
    with pytest.raises(SystemExit, match="needs --s3-endpoint"):
        main(["duckdb", "s3://b/docs", "filter_edu", "1g", str(tmp_path / "o"), str(tmp_path / "s")])
    cred = tmp_path / "c.env"
    cred.write_text("AWS_ACCESS_KEY_ID=u\nAWS_SECRET_ACCESS_KEY=p\n")
    apply_cred_file(str(cred))
    cred.write_text("AWS_ACCESS_KEY_ID=u\nEVIL=k\n")
    with pytest.raises(SystemExit, match="not an accepted S3 credential"):
        apply_cred_file(str(cred))


# ------------------------------------------------------------------ checksums
def test_checksum_and_id_set_comparison(tiny, tmp_path):
    from krishiv_bench.remote_filter_child import main

    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["duckdb", str(tiny), "filter_edu", "1g", str(a), str(a / "s")]) == 0
    assert main(["duckdb", str(tiny), "filter_edu", "1g", str(b), str(b / "s")]) == 0
    assert remote_filter.compare_results(a, b) is None
    (b / "result.parquet").unlink()
    pq.write_table(
        pa.table({"id": ["x"], "text": ["y"], "token_count": [1], "int_score": [5], "language_score": [0.99]}),
        b / "result.parquet",
    )
    diff = remote_filter.compare_results(a, b)
    assert diff is not None
    assert "n:" in diff or "id set" in diff or "sum_token" in diff


# ------------------------------------------------------------------ cap machinery (fresh scope; engineered child)
class FakePopen(FakeProc):
    """A Popen stand-in that drops ``stdout_text`` into the child's stdout file, then exits."""

    def __init__(self, *argv, stdout=None, **kw):
        super().__init__(alive=1, returncode=kw.get("rc", 0))
        self.pid = 424242
        if stdout is not None and kw.get("result") is not None:
            stdout.write(kw["result"])


def _patched_run(monkeypatch, result: bytes | None, rc: int, ooms: int):
    import krishiv_bench.capped as capped

    def popen(*a, **k):
        return FakePopen(*a, **k, result=result, rc=rc)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(capped, "find_cgroup", lambda pid, wait_s=5.0, _resolve=None: FakeCgroup([100], [ooms], [50]))


def test_capped_run_records_a_finished_child_with_metrics(workdir, monkeypatch, tiny):
    from krishiv_bench import capped

    assert capped.scope_command(None, ["x"])[6] == "MemoryMax=max"
    _patched_run(monkeypatch, b'RESULT {"version": "9", "setup_s": 0.1, "query_s": 0.5, "rows_out": 7}\n', 0, 0)
    rec = remote_filter.run_one("duckdb", tiny, "1g", 4 << 30, 60, workdir / "w")
    assert rec["status"] == "finished"
    assert (rec["rows_out"], rec["version"]) == (7, "9")
    assert rec["wall_s"] >= 0
    assert rec["fetched_bytes"] is None  # file reads traverse no object store
    assert "error" not in rec


def test_capped_run_records_a_killed_child(workdir, monkeypatch, tiny):
    _patched_run(monkeypatch, None, -9, 1)
    rec = remote_filter.run_one("duckdb", tiny, "1g", 4 << 30, 60, workdir / "w")
    assert rec["status"] == "killed"
    assert rec["oom_kills"] == 1


def test_cred_file_is_removed_and_never_recorded(workdir, monkeypatch, tiny):
    import krishiv_bench.capped as capped

    seen: dict = {}

    def popen(*a, **k):
        seen["argv"] = a[0]
        return FakePopen(*a, **k, result=b'RESULT {"version": "9", "setup_s": 0.1, "query_s": 0.5, "rows_out": 7}\n')

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(capped, "find_cgroup", lambda pid, wait_s=5.0, _resolve=None: FakeCgroup([100], [0], [50]))
    s3 = {"host": "127.0.0.1", "port": 19999, "bucket": "remote-filter", "user": "UU", "password": "PW-S3CR3T"}
    rec = remote_filter.run_one("duckdb", tiny, "1g", 4 << 30, 60, workdir / "w", mode="s3", s3=s3)
    assert rec["status"] == "finished"
    assert not (workdir / "w" / "s3creds.env").exists()
    assert "PW-S3CR3T" not in " ".join(seen["argv"])  # the secret is in the cred file, never on argv
    assert "PW-S3CR3T" not in json.dumps(rec)


def test_abba_order_and_caps():
    assert remote_filter.abba_order(["a", "b"], 0) == ["a", "b"]
    assert remote_filter.abba_order(["a", "b"], 1) == ["b", "a"]
    assert remote_filter.abba_order(["a", "b"], 2) == ["b", "a"]
    assert remote_filter.abba_order(["a", "b"], 3) == ["a", "b"]
    assert remote_filter.abba_order(["a", "b"], 4) == ["a", "b"]
    assert remote_filter.parse_caps("4g") == [("4g", 4 << 30), ("none", None)]
    assert remote_filter.parse_caps("none") == [("none", None)]
    ci = remote_filter.median_ci([1.0, 1.0, 1.0, 1.0, 1.0])
    assert ci == [1.0, 1.0]
    assert remote_filter.median_ci([]) is None


# ------------------------------------------------------------------ S3 scaffolding without docker
def test_container_command_is_throwaway_and_secret_free(tmp_path):
    cmd = remote_filter.s3_container_command(20001, tmp_path / "s", tmp_path / "e")
    assert isinstance(cmd, list)
    assert "--rm" in cmd
    assert "--name" not in cmd
    assert remote_filter.MINIO_IMAGE in cmd
    assert cmd[cmd.index("-p") + 1].startswith("127.0.0.1:")  # loopback only
    port1, (u1, p1) = remote_filter.random_port(), remote_filter.new_s3_credentials()
    port2, (u2, p2) = remote_filter.random_port(), remote_filter.new_s3_credentials()
    assert (port1, u1, p1) != (port2, u2, p2)  # random port and credentials per server
    env = tmp_path / "minio.env"
    remote_filter.write_env_file(env, u1, p1)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert p1 in env.read_text()
    # Creds travel in the env file, never on the command line.
    assert u1 not in cmd
    assert p1 not in cmd


def _origin_server(body: bytes):
    class Handler(BaseHTTPRequestHandler):
        def _serve(self):
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_PUT = _serve

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_counting_proxy_counts_response_bytes_and_keeps_host():
    body = b"x" * 10_000
    origin = _origin_server(body)
    proxy = CountingProxy("127.0.0.1", origin.server_port)
    try:
        with urllib.request.urlopen(proxy.url + "/docs/a.parquet") as r:
            assert r.read() == body
        assert proxy.response_bytes == len(body)
        data = b"y" * 500
        req = urllib.request.Request(proxy.url + "/docs/b.parquet", data=data, method="PUT")
        with urllib.request.urlopen(req) as r:
            assert r.read() == body
        assert proxy.response_bytes == 2 * len(body)
        echoed, netloc = _echoed_host()
        assert echoed == netloc  # the proxy keeps Host, so SigV4 signatures stay valid
    finally:
        proxy.close()
        origin.shutdown()
        origin.server_close()


def _echoed_host() -> tuple[str, str]:
    """The Host header an echo origin sees through a fresh proxy, and the proxy's own netloc."""
    seen: list[str] = []

    class Echo(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers["Host"])
            payload = self.headers["Host"].encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            pass

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    try:
        inner = CountingProxy("127.0.0.1", origin.server_port)
        try:
            with urllib.request.urlopen(inner.url + "/x") as r:
                assert r.read().decode() == seen[0]
        finally:
            inner.close()
    finally:
        origin.shutdown()
        origin.server_close()
    return seen[0], urllib.parse.urlsplit(inner.url).netloc


def test_streaming_put_and_get_roundtrip_without_loading_shards(tmp_path):
    import krishiv_bench.remote_filter_data as data_mod

    store: dict[str, bytes] = {}

    class Storage(BaseHTTPRequestHandler):
        def do_PUT(self):
            store[self.path] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            payload = store[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            pass

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Storage)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    try:
        endpoint = f"http://127.0.0.1:{origin.server_port}"
        src = tmp_path / "f.parquet"
        src.write_bytes(b"z" * (1 << 20))
        remote_filter.s3_put_file(endpoint, "b", "docs/f.parquet", src, "U", "P")
        assert remote_filter.s3_get_sha256(endpoint, "b", "docs/f.parquet", "U", "P") == data_mod.sha256_file(src)
    finally:
        origin.shutdown()
        origin.server_close()


def test_sigv4_signing_is_deterministic_and_secret_scoped():
    def sign(password: str) -> dict[str, str]:
        return remote_filter.s3_authorization(
            "U",
            password,
            method="GET",
            path="/b/docs/a.parquet",
            payload=b"",
            host="127.0.0.1:9000",
            amz_date="20261027T000000Z",
        )

    h1, h2, h3 = sign("SECRET-1"), sign("SECRET-1"), sign("SECRET-2")
    assert h1 == h2
    assert h1 != h3
    assert h1["authorization"].startswith("AWS4-HMAC-SHA256 Credential=U/20261027/us-east-1/s3/aws4_request")
    assert "SECRET-1" not in h1["authorization"]
    with pytest.raises(OSError, match=r"S3 GET /b/docs/a\.parquet") as exc_info:
        remote_filter.s3_request("http://127.0.0.1:1", "b", "docs/a.parquet", "U", "SECRET-1")
    assert "SECRET-1" not in str(exc_info.value)  # failures name method/path/status, never the secret


# ------------------------------------------------------------------ orchestration with scripted runs
class Script:
    """Replaces ``run_one``: engine behaviour as a list of statuses, one per call."""

    def __init__(self, plan: dict[str, list[str]], rows: dict[str, int] | None = None):
        self.plan, self.rows, self.calls = {k: list(v) for k, v in plan.items()}, rows or {}, []

    def __call__(self, engine, data, engine_memory, cap_bytes, timeout_s, work: Path, **kw) -> dict:
        self.calls.append((engine, kw.get("mode", "file")))
        status = self.plan[engine].pop(0) if len(self.plan[engine]) > 1 else self.plan[engine][0]
        (work / "out").mkdir(parents=True, exist_ok=True)
        rec = {
            "engine": engine,
            "query": "filter_edu",
            "status": status,
            "wall_s": 2.0,
            "peak_rss_bytes": 1 << 30,
            "spill_bytes": 0,
            "fetched_bytes": 11,
            "returncode": 0,
            "oom_kills": int(status == "killed"),
        }
        if status == "finished":
            rec |= {"version": "1", "setup_s": 0.1, "query_s": 1.0, "rows_out": self.rows.get(engine, 10)}
        return rec


def fake_check(results: dict[str, str | None]):
    def check(ref_out: Path, got_out: Path) -> str | None:
        engine = got_out.parent.name.split("-")[-1]
        return results[engine]

    return check


@pytest.fixture
def env(workdir: Path, tiny: Path) -> Path:
    return tiny


def run_suite(env, script, check, engines, reps=5, **kw):
    f = _run_real(env, "file", "4g", engines, "3g", reps, None, script, check, serve=_NoServer, **kw)
    return json.loads(f.read_text())


class _NoServer:
    def __init__(self, *a, **k):
        raise AssertionError("file mode starts no server")


def test_validation_precedes_timing_and_a_wrong_result_is_never_timed(env, workdir):
    script = Script({"duckdb": ["finished"], "polars": ["finished"]})
    r = run_suite(env, script, fake_check({"polars": None}), ["duckdb", "polars"], reps=5)
    q = r["scopes"]["4g"]["engines"]
    assert q["duckdb"]["verdict"] == "reference"
    assert q["polars"]["verdict"] == "ok"
    assert [x["phase"] for x in q["polars"]["runs"]] == ["validation"] + ["timed"] * 5
    assert script.calls.count(("polars", "file")) == 12  # validation + 5 timed, in each of the 2 scopes
    assert r["scopes"]["none"]["engines"]["polars"]["verdict"] == "ok"
    assert q["polars"]["summary"]["median_wall_s"] == 2.0
    assert q["polars"]["summary"]["wall_ci95"] == [2.0, 2.0]
    assert q["polars"]["summary"]["n"] == 5


def test_a_wrong_result_is_never_timed(env, workdir):
    script = Script({"duckdb": ["finished"], "polars": ["finished"]})
    r = run_suite(env, script, fake_check({"polars": "n: reference 10, got 11"}), ["duckdb", "polars"], reps=5)
    q = r["scopes"]["4g"]["engines"]
    assert q["polars"]["verdict"].startswith("mismatch: n:")
    assert script.calls.count(("polars", "file")) == 2  # one validation run per scope, never timed
    assert q["polars"]["summary"]["n"] == 0
    assert q["polars"]["runs"][0]["status"] == "finished"  # the validation result itself, verdict mismatch


def test_killed_runs_are_counted_and_checksum_errors_stay_unverified(env, workdir):
    script = Script({"duckdb": ["finished"], "polars": ["killed"]})
    r = run_suite(env, script, fake_check({"polars": None}), ["duckdb", "polars"], reps=5)
    p = r["scopes"]["4g"]["engines"]["polars"]
    assert p["verdict"] == "unverified: validation run killed"
    assert p["summary"]["finished"] == 0
    assert p["summary"]["killed"] == 5

    def boom(ref_out, got_out):
        raise RuntimeError("no parquet")

    r = run_suite(env, Script({"duckdb": ["finished"], "polars": ["finished"]}), boom, ["duckdb", "polars"], reps=5)
    assert r["scopes"]["4g"]["engines"]["polars"]["verdict"] == "unverified: RuntimeError: no parquet"


def test_input_and_engine_validation(env, workdir):
    s = Script({"duckdb": ["finished"]})
    with pytest.raises(ValueError, match="--reps must be at least 1"):
        remote_filter.run_remote_filter(env, "file", "4g", ["duckdb"], "3g", 0, None, s)
    with pytest.raises(ValueError, match="engines must be a non-empty subset"):
        remote_filter.run_remote_filter(env, "file", "4g", [], "3g", 5, None, s)
    with pytest.raises(ValueError, match="s3 mode needs the server"):
        remote_filter.run_one("duckdb", env, "3g", 4 << 30, None, workdir / "w9", mode="s3", s3=None)
    with pytest.raises(ValueError, match="unknown remote-filter mode"):
        remote_filter.run_remote_filter(env, "filey", "4g", ["duckdb"], "3g", 5, None, s)
    with pytest.raises(NotImplementedError, match="later unit"):
        remote_filter.run_remote_filter(env, "https", "4g", ["duckdb"], "3g", 5, None, s)
    with pytest.raises(ValueError, match="memory size must"):
        remote_filter.run_remote_filter(env, "file", "lots", ["duckdb"], "3g", 5, None, s)
    with pytest.raises(FileNotFoundError, match="krishiv-datasets remote-filter"):
        remote_filter.run_remote_filter(workdir / "nodata", "file", "4g", ["duckdb"], "3g", 5, None, s)


def test_s3_mode_records_server_facts_without_secrets(env, workdir, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "POISON-SECRET")
    seen: dict = {}

    class FakeServer:
        image = remote_filter.MINIO_IMAGE

        def __init__(self, *a, **k):
            self.port, self.user, self.password = 23456, "UU", "PW-S3CR3T"

        def start(self):
            return self

        @property
        def endpoint(self):
            return "http://127.0.0.1:23456"

        def seed(self, docs_dir):
            return ["docs/a.parquet"]

        def verify_seed(self, docs_dir):
            seen["verified"] = True

        def stop(self):
            seen["stopped"] = True

    def fake_run(engine, data, engine_memory, cap_bytes, timeout_s, work, **kw):
        assert kw["s3"]["password"] == "PW-S3CR3T"  # secrets travel by argument, never in the record
        (work / "out").mkdir(parents=True, exist_ok=True)
        return {
            "engine": engine,
            "status": "finished",
            "wall_s": 2.0,
            "peak_rss_bytes": 1,
            "spill_bytes": 0,
            "fetched_bytes": 99,
            "returncode": 0,
            "oom_kills": 0,
            "version": "1",
            "setup_s": 0.1,
            "query_s": 1.0,
            "rows_out": 10,
        }

    f = remote_filter.run_remote_filter(
        env, "s3", "4g", ["duckdb"], "3g", 5, None, fake_run, fake_check({}), serve=FakeServer
    )
    r = json.loads(f.read_text())
    assert seen == {"verified": True, "stopped": True}
    assert r["s3"] == {
        "endpoint_host": "127.0.0.1",
        "bucket": "remote-filter",
        "image": remote_filter.MINIO_IMAGE,
        "keys": 1,
    }
    f = next(iter((workdir / "results").iterdir()))
    text = f.read_text()
    assert "PW-S3CR3T" not in text
    assert "POISON-SECRET" not in text
    assert r["scopes"]["4g"]["engines"]["duckdb"]["summary"]["median_fetched_bytes"] == 99


def test_report_table_from_results(env, workdir):
    script = Script({"duckdb": ["finished"], "polars": ["killed"]})
    run_suite(env, script, fake_check({"polars": None}), ["duckdb", "polars"], reps=5)
    md = remote_filter.render_remote_filter("file")
    assert "| duckdb | 4g | 3g | 2.00 (2.00-2.00) | 5/5 | 1.00 |" in md
    assert "null" not in md  # file mode shows `-` for fetched bytes, with the footnote saying why
    assert "no engine-independent count" in md
    assert remote_filter.write_remote_filter_report("file").name == "report_remotefilter_file.md"
    with pytest.raises(SystemExit, match="no remote-filter results"):
        remote_filter.render_remote_filter("s3")


# ------------------------------------------------------------------ command line
def test_cli_run_reproduce_and_report_remote_filter(workdir, tiny, monkeypatch, capsys):
    from krishiv_bench import cli
    from krishiv_bench import remote_filter as rf

    calls: list[tuple] = []
    monkeypatch.setattr(rf, "run_remote_filter", lambda *a, **k: calls.append((a, k)))
    cli.main(
        [
            "run",
            "--engine",
            "duckdb",
            "--suite",
            "remote-filter",
            "--input",
            str(tiny),
            "--cap",
            "4g",
            "--iterations",
            "5",
        ]
    )
    assert calls[0][0][:5] == (tiny, "file", "4g", ["duckdb"], "3g")
    assert calls[0][0][5] == 5
    calls.clear()
    cli.main(
        ["reproduce", "--suite", "remote-filter", "--engines", "krishiv,duckdb", "--input", str(tiny), "--reps", "5"]
    )
    assert calls[0][0][3] == ["krishiv", "duckdb"]
    assert calls[0][0][2] == "4g"  # the shared flag defaults to memcap's 4.5g; the suite scope is 4 GiB
    with pytest.raises(SystemExit) as e:
        cli.main(["reproduce", "--suite", "remote-filter", "--reps", "2", "--input", str(tiny)])
    assert e.value.code == 2
    assert "at least 5 repetitions" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--engine", "datafusion", "--suite", "remote-filter", "--input", str(tiny)])
    assert "remote-filter engines are" in capsys.readouterr().err
    run_suite(tiny, Script({"duckdb": ["finished"]}), fake_check({}), ["duckdb"], reps=5)
    cli.main(["report", "--suite", "remote-filter", "--mode", "file"])
    assert "report_remotefilter_file.md" in capsys.readouterr().out


def test_datasets_cli_remote_filter(workdir, monkeypatch, tmp_path, capsys):
    from krishiv_bench import datasets

    seen = []
    monkeypatch.setattr(remote_filter_data, "fetch", lambda mirror=None: seen.append(mirror))
    datasets.main(["remote-filter"])
    datasets.main(["remote-filter", "--mirror", str(tmp_path)])
    assert seen == [None, tmp_path]
    monkeypatch.setattr(remote_filter_data, "fetch", lambda mirror=None: (_ for _ in ()).throw(OSError("net")))
    with pytest.raises(SystemExit):
        datasets.main(["remote-filter"])
    assert "net" in capsys.readouterr().err


def test_scope_command_max_and_sizes():
    from krishiv_bench import capped

    assert "MemoryMax=max" in capped.scope_command(None, ["x"])
    assert "MemoryMax=4096M" in capped.scope_command(4 << 30, ["x"])
