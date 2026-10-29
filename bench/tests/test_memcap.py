"""memcap suite: generator determinism, queries against a DuckDB oracle, kill classification, orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from krishiv_bench import memcap, memcap_data
from krishiv_bench.memcap import Observation, classify, compare_checksums, monitor, resolve_queries, rotated
from krishiv_bench.memcap_queries import CHECKS, QUERIES, SQL, polars_query, sql_for
from krishiv_bench.sizes import parse_size

GIB = 1 << 30
CAP = parse_size("4.5g")
TINY_ROWS = 20_000
TINY_CHUNK = 8_000

# Content digests of the generator at TINY_ROWS / TINY_CHUNK. A change here means the generated rows changed:
# that is a new dataset version (``VERSION``), never an edit of these constants.
TINY_SALES_SHA256 = "c37c506662de22f25d5bfc10de402c871d9d8b663c347ec52fe3d7a1c1d37ed1"
TINY_PRODUCTS_SHA256 = "97de480426c299dcad9cf14383a83feb906b2b113728dbd774dd2e4bf7c58c8b"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("memcap") / "tiny"
    return out, memcap_data.gen_memcap(TINY_ROWS, out, chunk=TINY_CHUNK)


def _sql(text: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect().execute(text)


# ------------------------------------------------------------------ generator
def test_generator_is_deterministic(tiny, tmp_path):
    _out, rec = tiny
    again = memcap_data.gen_memcap(TINY_ROWS, tmp_path / "again", chunk=TINY_CHUNK)
    assert again["sales_content_sha256"] == rec["sales_content_sha256"]
    assert again["products_content_sha256"] == rec["products_content_sha256"]
    assert rec["sales_content_sha256"] == TINY_SALES_SHA256
    assert rec["products_content_sha256"] == TINY_PRODUCTS_SHA256


def test_digest_changes_with_the_chunking_and_matches_the_files(tiny, tmp_path):
    out, rec = tiny
    other = memcap_data.gen_memcap(TINY_ROWS, tmp_path / "other", chunk=TINY_CHUNK + 1)
    assert other["sales_content_sha256"] != rec["sales_content_sha256"]
    assert json.loads((out / "digest.json").read_text()) == rec
    assert memcap_data.load_digest(out) == rec


def test_schema_and_distributions(tiny):
    out, rec = tiny
    assert rec["sales_files"] == 3  # 8000 + 8000 + 4000
    t = pq.read_table(out / "sales")
    assert t.num_rows == TINY_ROWS
    assert t.column_names == ["order_id", "ts", "customer_id", "product_id", "store_id", "qty", "price", "status"]
    assert [str(f.type) for f in t.schema] == [
        "int64",
        "timestamp[ms]",  # Parquet has no second unit
        "int32",
        "int32",
        "int16",
        "int16",
        "double",
        "string",
    ]
    stats = _sql(
        f"""SELECT count(DISTINCT order_id), max(customer_id), max(product_id), max(store_id), min(qty), max(qty),
                   min(price), max(price), count(DISTINCT status),
                   avg((status = 'completed')::INT)
            FROM read_parquet('{out}/sales/*.parquet')"""
    ).fetchall()[0]
    assert stats[0] == TINY_ROWS
    assert stats[1] < TINY_ROWS // 10
    assert stats[2] < 50_000
    assert stats[3] < 500
    assert (stats[4], stats[5]) == (1, 9)
    assert stats[6] >= 1.0
    assert stats[7] <= 500.0
    assert stats[8] == 5
    assert stats[9] == pytest.approx(0.55, abs=0.02)
    p = pq.read_table(out / "products.parquet")
    assert p.num_rows == 50_000
    assert p.column_names == ["product_id", "category", "brand", "list_price"]


def test_existing_dataset_is_never_overwritten(tiny):
    out, _ = tiny
    before = sorted(p.name for p in out.rglob("*"))
    with pytest.raises(FileExistsError, match="never regenerated in place"):
        memcap_data.gen_memcap(TINY_ROWS, out, chunk=TINY_CHUNK)
    assert sorted(p.name for p in out.rglob("*")) == before


def test_failed_generation_leaves_no_directory(tmp_path):
    with pytest.raises(ValueError, match="chunk must be positive"):
        memcap_data.gen_memcap(10, tmp_path / "x", chunk=0)
    with pytest.raises(ValueError, match="rows must be positive"):
        memcap_data.gen_memcap(0, tmp_path / "x")
    assert not list(tmp_path.iterdir())


def test_rows_labels_and_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    assert memcap_data.parse_rows("150m") == 150_000_000
    assert memcap_data.rows_label(10_000_000) == "10m"
    assert memcap_data.memcap_dir("50M") == tmp_path / "memcap" / "v1" / "50m"
    assert memcap_data.n_customers(150_000_000) == 15_000_000
    for bad in ("0m", "10", "1.5m", "m", "-5m"):
        with pytest.raises(ValueError, match="whole number of millions"):
            memcap_data.parse_rows(bad)
    with pytest.raises(ValueError, match="multiple of 1,000,000"):
        memcap_data.rows_label(1500)
    with pytest.raises(FileNotFoundError, match="krishiv-datasets memcap --rows 10m"):
        memcap_data.load_digest(tmp_path / "10m")


def test_generate_writes_manifest_and_skips_existing(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    monkeypatch.setattr(memcap_data, "CHUNK", 1_000_000)
    monkeypatch.setattr(memcap_data, "gen_memcap", lambda rows, out: _fake_gen(rows, out))
    d = memcap_data.generate("1m")
    assert d == tmp_path / "memcap" / "v1" / "1m"
    assert (tmp_path / "manifests" / "memcap-v1-1m.sha256").read_text().strip().endswith("digest.json")
    assert memcap_data.generate("1m") == d
    assert "not regenerating" in capsys.readouterr().out


def _fake_gen(rows: int, out: Path) -> dict:
    out.mkdir()
    (out / "digest.json").write_text("{}")
    return {"sales_content_sha256": "0" * 64}


# ------------------------------------------------------------------ queries against an oracle
def _oracle(out: Path, q: str) -> list[tuple]:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW sales AS SELECT * FROM read_parquet('{out}/sales/*.parquet')")
    con.execute(f"CREATE VIEW products AS SELECT * FROM read_parquet('{out}/products.parquet')")
    return con.execute(SQL[q]).fetchall()


def test_q1_matches_a_handwritten_oracle(tiny):
    out, _ = tiny
    expected = _sql(
        f"""SELECT store_id, status, count(*), sum(qty * price)
            FROM read_parquet('{out}/sales/*.parquet')
            WHERE ts >= '2026-06-01' AND status IN ('completed', 'shipped') GROUP BY ALL"""
    ).fetchall()
    got = _oracle(out, "q1_filter_agg")
    assert len(got) > 0
    assert [r[:3] for r in sorted(got)] == [r[:3] for r in sorted(expected)]
    assert [r[3] for r in sorted(got)] == pytest.approx([r[3] for r in sorted(expected)])


def test_q2_group_count_and_totals(tiny):
    out, _ = tiny
    rows = _oracle(out, "q2_join_agg")
    kept = _sql(f"SELECT count(*) FROM read_parquet('{out}/sales/*.parquet') WHERE status <> 'cancelled'")
    assert sum(r[2] for r in rows) == kept.fetchall()[0][0]
    assert {r[0] for r in rows} <= set(memcap_data.CATEGORIES.tolist())


def test_q3_keeps_the_latest_order_of_every_customer(tiny):
    out, _ = tiny
    rows = _oracle(out, "q3_window_latest")
    distinct = _sql(f"SELECT count(DISTINCT customer_id) FROM read_parquet('{out}/sales/*.parquet')")
    assert len(rows) == distinct.fetchall()[0][0]
    expected = _sql(
        f"""SELECT customer_id, order_id, ts FROM read_parquet('{out}/sales/*.parquet') s
            WHERE NOT EXISTS (
                SELECT 1 FROM read_parquet('{out}/sales/*.parquet') s2
                WHERE s2.customer_id = s.customer_id
                  AND (s2.ts > s.ts OR (s2.ts = s.ts AND s2.order_id > s.order_id))
            )"""
    ).fetchall()
    assert sorted((r[0], r[1], r[2]) for r in rows) == sorted((r[0], r[1], r[2]) for r in expected)


def test_q4_one_row_per_customer(tiny):
    out, _ = tiny
    rows = _oracle(out, "q4_highcard_agg")
    assert sum(r[1] for r in rows) == TINY_ROWS
    assert len({r[0] for r in rows}) == len(rows)


def test_checks_run_on_every_query_result_and_agree_with_polars(tiny, tmp_path):
    pl = pytest.importorskip("polars")
    out, _ = tiny
    for q in QUERIES:
        res = tmp_path / q
        res.mkdir()
        con = duckdb.connect()
        con.execute(f"CREATE VIEW sales AS SELECT * FROM read_parquet('{out}/sales/*.parquet')")
        con.execute(f"CREATE VIEW products AS SELECT * FROM read_parquet('{out}/products.parquet')")
        con.execute(f"COPY ({SQL[q]}) TO '{res}/duck.parquet' (FORMAT parquet)")
        pol = res / "polars"
        pol.mkdir()
        polars_query(
            q, pl.scan_parquet(f"{out}/sales/*.parquet"), pl.scan_parquet(f"{out}/products.parquet")
        ).sink_parquet(pol / "result.parquet")
        ref = _checksum_file(res / "duck.parquet", q)
        got = memcap.checksum(pol, q)
        assert compare_checksums(ref, got) is None, q
        assert ref["n"] > 0


def _checksum_file(f: Path, q: str) -> dict:
    d = f.parent / "only"
    d.mkdir(exist_ok=True)
    (d / f.name).write_bytes(f.read_bytes())
    return memcap.checksum(d, q)


def test_checksum_comparison():
    ref = {"n": 3, "revenue": 1.0e10}
    assert compare_checksums(ref, {"n": 3, "revenue": 1.0e10 * (1 + 1e-12)}) is None
    assert "n: reference 3, got 4" in (compare_checksums(ref, {"n": 4, "revenue": 1.0e10}) or "")
    assert "revenue" in (compare_checksums(ref, {"n": 3, "revenue": 1.0e10 + 100}) or "")
    assert "revenue" in (compare_checksums(ref, {"n": 3, "revenue": None}) or "")
    assert "n: reference 3, got None" in (compare_checksums(ref, {"revenue": 1.0e10}) or "")


# ------------------------------------------------------------------ kill / OOM classification
class FakeProc:
    """A ``Popen`` stand-in: ``poll()`` returns None for ``alive`` calls, then ``returncode``."""

    def __init__(self, alive: int, returncode: int | None):
        self.alive, self._rc, self.returncode, self.killed = alive, returncode, None, False

    def poll(self):
        if self.killed:
            return self.returncode
        if self.alive > 0:
            self.alive -= 1
            return None
        self.returncode = self._rc
        return self._rc

    def kill(self):
        self.killed, self.returncode = True, -9

    def wait(self):
        return self.returncode


class FakeCgroup:
    def __init__(
        self,
        peaks: list[int | None],
        ooms: list[int | None],
        rss: list[int | None],
        hits: list[int | None] | None = None,
    ):
        self.peaks, self.ooms, self.rsss, self.kills = peaks, ooms, rss, 0
        self.hits = hits if hits is not None else [0]

    @staticmethod
    def _next(xs: list[int | None]) -> int | None:
        return xs.pop(0) if len(xs) > 1 else (xs[0] if xs else None)

    def peak(self):
        return self._next(self.peaks)

    def oom_kills(self):
        return self._next(self.ooms)

    def limit_hits(self):
        return self._next(self.hits)

    def rss(self):
        return self._next(self.rsss)

    def kill_all(self):
        self.kills += 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def obs(
    returncode: int | None = 0,
    timed_out: bool = False,
    peak_bytes: int = GIB,
    oom_kills: int = 0,
    limit_hits: int = 0,
) -> Observation:
    return Observation(returncode, timed_out, peak_bytes, oom_kills, None, 1.0, 0, limit_hits)


@pytest.mark.parametrize(
    ("kw", "got_result", "want"),
    [
        ({}, True, "finished"),
        ({}, False, "failed"),  # exited 0 but printed nothing
        ({"returncode": 1}, False, "failed"),
        ({"returncode": 1, "peak_bytes": CAP}, False, "failed"),  # an ordinary error near the cap is not a kill
        ({"returncode": -9, "oom_kills": 1, "peak_bytes": GIB}, False, "killed"),
        ({"returncode": 1, "oom_kills": 2}, False, "killed"),  # the parent survived a killed JVM child
        ({"returncode": 0, "oom_kills": 1}, True, "killed"),  # a counted kill wins over a clean exit
        ({"returncode": -9, "peak_bytes": CAP}, False, "killed"),  # event missed: SIGKILL with the scope full
        ({"returncode": 137, "peak_bytes": int(0.96 * CAP)}, False, "killed"),
        ({"returncode": -9, "peak_bytes": int(0.5 * CAP)}, False, "failed"),  # SIGKILL from elsewhere
        # Killed between two polls: the sampled peak stayed low, but an earlier poll saw the scope hit its limit.
        ({"returncode": -9, "peak_bytes": int(0.5 * CAP), "limit_hits": 3}, False, "killed"),
        ({"returncode": 1, "peak_bytes": int(0.5 * CAP), "limit_hits": 3}, False, "failed"),  # not a SIGKILL
        ({"returncode": 0, "limit_hits": 3}, True, "finished"),  # page-cache pressure alone is not a kill
        ({"returncode": -9, "timed_out": True, "oom_kills": 1}, False, "timeout"),
    ],
)
def test_classify(kw, got_result, want):
    assert classify(obs(**kw), CAP, got_result) == want


def test_monitor_records_peaks_and_the_oom_kill_of_a_dying_process():
    clock = Clock()
    proc = FakeProc(alive=4, returncode=-9)
    cg = FakeCgroup(peaks=[10, 400, 4_800, None], ooms=[0, 0, 1, None], rss=[5, 300, 4_000, None])
    o = monitor(proc, cg, None, None, clock, clock.sleep)
    assert (o.returncode, o.peak_bytes, o.oom_kills, o.rss_peak_bytes) == (-9, 4_800, 1, 4_000)
    assert o.wall_s == pytest.approx(4 * memcap.POLL_S)
    assert cg.kills == 1  # stragglers are cleared after the process exits
    assert classify(o, 5_000, got_result=False) == "killed"


def test_monitor_kills_the_scope_on_timeout():
    clock = Clock()
    proc = FakeProc(alive=10**9, returncode=0)
    cg = FakeCgroup([1], [0], [1])
    o = monitor(proc, cg, None, 0.05, clock, clock.sleep)
    assert o.timed_out
    assert proc.killed
    assert cg.kills == 2  # once on the timeout, once at the end
    assert classify(o, CAP, got_result=False) == "timeout"


def test_monitor_samples_the_scratch_directory_peak(tmp_path):
    clock = Clock()
    sizes = iter([0, 7, 3, 3])
    proc = FakeProc(alive=3 * memcap.SPILL_EVERY, returncode=0)
    o = monitor(proc, None, tmp_path, None, clock, clock.sleep, lambda _p: next(sizes))
    assert o.spill_peak_bytes == 7
    assert o.peak_bytes == 0  # no cgroup found


def test_cgroup_reads_files_and_survives_removal(tmp_path):
    (tmp_path / "memory.peak").write_text("123\n")
    (tmp_path / "memory.current").write_text("200\n")
    (tmp_path / "memory.events").write_text("low 0\nhigh 0\nmax 4\noom 0\noom_kill 2\n")
    (tmp_path / "memory.stat").write_text("anon 100\nfile 900\nfile_mapped 20\nshmem 3\n")
    cg = memcap.Cgroup(tmp_path)
    assert (cg.peak(), cg.oom_kills(), cg.rss(), cg.limit_hits()) == (200, 2, 123, 4)
    cg.kill_all()
    assert (tmp_path / "cgroup.kill").read_text() == "1"
    gone = memcap.Cgroup(tmp_path / "missing")
    assert (gone.peak(), gone.oom_kills(), gone.rss(), gone.limit_hits()) == (None, None, None, None)
    (tmp_path / "memory.events").write_text("low 0\nhigh 0\noom 0\n")
    assert cg.limit_hits() is None  # no `max` line
    (tmp_path / "memory.events").write_text("max x\n")
    assert cg.limit_hits() is None  # unparsable
    gone.kill_all()  # no error


def test_scope_command_sets_the_cap_and_no_swap():
    cmd = memcap.scope_command(CAP, ["python", "-m", "x"])
    assert cmd[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "MemoryMax=4608M" in cmd
    assert "MemorySwapMax=0" in cmd
    assert cmd[-3:] == ["python", "-m", "x"]
    assert cmd[cmd.index("--") + 1 :][0] == "python"


def test_read_result_takes_the_last_result_line():
    out = 'warn\nRESULT {"a": 1}\nnoise\nRESULT {"a": 2}\n'
    assert memcap.read_result(out) == {"a": 2}
    assert memcap.read_result("RESULT {broken\n") is None
    assert memcap.read_result("nothing\n") is None


# ------------------------------------------------------------------ arguments
def test_sizes_and_query_names():
    assert parse_size("4.5g") == 4_831_838_208
    assert parse_size("1G") == 1 << 30
    assert parse_size("1024m") == 1 << 30
    for bad in ("", "0g", "-1g", "3", "3t", "g", "0.0001g"):
        with pytest.raises(ValueError, match="memory size must"):
            parse_size(bad)
    assert resolve_queries(["q1", "q3_window_latest", "q1_filter_agg"]) == ["q1_filter_agg", "q3_window_latest"]
    with pytest.raises(ValueError, match="unknown memcap query 'q9'"):
        resolve_queries(["q9"])
    assert rotated(["a", "b", "c"], 0) == ["a", "b", "c"]
    assert rotated(["a", "b", "c"], 1) == ["b", "c", "a"]
    assert rotated(["a", "b", "c"], 4) == ["b", "c", "a"]


# ------------------------------------------------------------------ orchestration with scripted runs
class Script:
    """Replaces ``run_one``: engine behaviour per (engine, query) as a list of statuses, one per call."""

    def __init__(self, plan: dict[str, list[str]], rows: dict[str, int] | None = None):
        self.plan, self.rows, self.calls = {k: list(v) for k, v in plan.items()}, rows or {}, []

    def __call__(self, engine, data, query, engine_memory, cap_bytes, timeout_s, work: Path) -> dict:
        self.calls.append(engine)
        status = self.plan[engine].pop(0) if len(self.plan[engine]) > 1 else self.plan[engine][0]
        (work / "out").mkdir(parents=True, exist_ok=True)
        rec = {"engine": engine, "query": query, "status": status, "wall_s": 2.0, "peak_rss_bytes": GIB}
        rec |= {"spill_bytes": 0, "returncode": 0, "oom_kills": int(status == "killed")}
        if status == "finished":
            rec |= {"version": "1", "setup_s": 0.1, "query_s": 1.0, "rows_out": self.rows.get(engine, 10)}
        return rec


def fake_check(results: dict[str, dict]):
    def check(out: Path, query: str) -> dict:
        engine = out.parent.name.split("-")[1]
        return results[engine]

    return check


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    d = memcap_data.memcap_dir("10m")
    d.mkdir(parents=True)
    (d / "digest.json").write_text(json.dumps({"sales_content_sha256": "ab" * 32, "rows": 10_000_000}))
    return tmp_path


def run_suite(env, script, check, engines, reps=2, queries=("q1_filter_agg",)):
    f = memcap.run_memcap("10m", "4.5g", engines, "3g", reps, list(queries), None, script, check)
    return json.loads(f.read_text())


def test_validation_precedes_timing_and_the_reference_runs_first(env):
    script = Script({"duckdb": ["finished"], "polars": ["finished"]})
    ok = {"n": 10, "revenue": 5.0}
    r = run_suite(env, script, fake_check({"duckdb": ok, "polars": ok}), ["polars", "duckdb"])
    assert script.calls == [
        "duckdb",
        "polars",
        "duckdb",
        "polars",
        "polars",
        "duckdb",
    ]  # validation, rep 0, rep 1 rotated
    q = r["queries"]["q1_filter_agg"]["engines"]
    assert [x["phase"] for x in q["polars"]["runs"]] == ["validation", "timed", "timed"]
    assert q["duckdb"]["verdict"] == "reference"
    assert q["polars"]["verdict"] == "ok"
    assert q["polars"]["summary"] | {} == {
        "n": 2,
        "finished": 2,
        "killed": 0,
        "median_wall_s": 2.0,
        "median_peak_rss_bytes": GIB,
        "max_peak_rss_bytes": GIB,
        "median_spill_bytes": 0,
    }


def test_a_wrong_result_is_never_timed(env):
    script = Script({"duckdb": ["finished"], "polars": ["finished"]})
    check = fake_check({"duckdb": {"n": 10, "revenue": 5.0}, "polars": {"n": 10, "revenue": 6.0}})
    r = run_suite(env, script, check, ["duckdb", "polars"])
    q = r["queries"]["q1_filter_agg"]["engines"]
    assert q["polars"]["verdict"].startswith("mismatch: revenue")
    assert script.calls.count("polars") == 1  # the validation run only
    assert q["polars"]["summary"]["n"] == 0
    assert q["duckdb"]["summary"]["finished"] == 2


def test_killed_runs_are_counted_failed_and_the_engine_is_still_attempted(env):
    script = Script({"duckdb": ["finished"], "polars": ["killed"]})
    r = run_suite(env, script, fake_check({"duckdb": {"n": 10}}), ["duckdb", "polars"], reps=3)
    p = r["queries"]["q1_filter_agg"]["engines"]["polars"]
    assert p["verdict"] == "unverified: validation run killed"
    assert p["summary"]["finished"] == 0
    assert p["summary"]["killed"] == 3
    assert p["summary"]["n"] == 3
    assert p["summary"]["median_wall_s"] is None


def test_an_engine_that_failed_validation_is_verified_on_its_first_finished_run(env):
    script = Script({"duckdb": ["finished"], "polars": ["killed", "finished"]})
    r = run_suite(env, script, fake_check({"duckdb": {"n": 10}, "polars": {"n": 10}}), ["duckdb", "polars"], reps=2)
    p = r["queries"]["q1_filter_agg"]["engines"]["polars"]
    assert p["verdict"] == "ok"
    assert p["summary"]["finished"] == 2  # both timed runs finished (the killed one was the validation run)


def test_row_count_drift_after_validation_marks_a_mismatch(env):
    script = Script({"duckdb": ["finished"], "polars": ["finished"]}, rows={"polars": 11})
    r = run_suite(env, script, fake_check({"duckdb": {"n": 10}, "polars": {"n": 10}}), ["duckdb", "polars"], reps=2)
    p = r["queries"]["q1_filter_agg"]["engines"]["polars"]
    assert p["verdict"].startswith("mismatch: n")
    assert p["summary"]["finished"] == 0


def test_without_a_reference_result_engines_are_unverified_not_dropped(env):
    script = Script({"duckdb": ["killed"], "polars": ["finished"]})
    r = run_suite(env, script, fake_check({"polars": {"n": 10}}), ["polars"], reps=1)
    q = r["queries"]["q1_filter_agg"]["engines"]
    assert q["polars"]["verdict"] == "unverified: no reference result"
    assert q["polars"]["summary"]["finished"] == 1
    assert r["reported_engines"] == ["polars"]  # duckdb ran only as the reference


def test_checker_errors_are_unverified(env):
    def boom(out, q):
        raise RuntimeError("no parquet")

    r = run_suite(env, Script({"duckdb": ["finished"]}), boom, ["duckdb"], reps=1)
    assert r["queries"]["q1_filter_agg"]["engines"]["duckdb"]["verdict"] == "unverified: RuntimeError: no parquet"


def test_input_validation(env):
    s = Script({"duckdb": ["finished"]})
    with pytest.raises(ValueError, match="--reps must be at least 1"):
        memcap.run_memcap("10m", "4.5g", ["duckdb"], "3g", 0, None, None, s)
    with pytest.raises(ValueError, match="engines must be a non-empty subset"):
        memcap.run_memcap("10m", "4.5g", ["flink"], "3g", 1, None, None, s)
    with pytest.raises(ValueError, match="unknown memcap queries"):
        memcap.run_memcap("10m", "4.5g", ["duckdb"], "3g", 1, ["q9"], None, s)
    with pytest.raises(ValueError, match="memory size must"):
        memcap.run_memcap("10m", "lots", ["duckdb"], "3g", 1, None, None, s)
    with pytest.raises(FileNotFoundError, match="krishiv-datasets memcap --rows 50m"):
        memcap.run_memcap("50m", "4.5g", ["duckdb"], "3g", 1, None, None, s)


def test_report_table_from_results(env):
    script = Script({"duckdb": ["finished"], "polars": ["killed"]})
    run_suite(env, script, fake_check({"duckdb": {"n": 10}}), ["duckdb", "polars"], reps=3)
    md = memcap.render_memcap("10m")
    assert "| q1_filter_agg | duckdb | 3g | 2.00 | 3/3 | 1.00 | 0.00 | reference |" in md
    assert "| q1_filter_agg | polars | 3g | - | 0/3 | >4.5 (killed) | - | unverified: validation run killed |" in md
    assert memcap.write_memcap_report("10m").name == "report_memcap_10m.md"
    with pytest.raises(SystemExit, match="no memcap results for 50m"):
        memcap.render_memcap("50m")


def test_report_keeps_the_newest_result_per_engine_and_memory(env):
    script = Script({"duckdb": ["finished"]})
    for mem in ("3g", "1g"):
        memcap.run_memcap(
            "10m", "4.5g", ["duckdb"], mem, 1, ["q1_filter_agg"], None, script, fake_check({"duckdb": {"n": 10}})
        )
    best = memcap.latest_memcap("10m")
    assert sorted(best) == [("duckdb", "1g"), ("duckdb", "3g")]


# ------------------------------------------------------------------ real child processes
@pytest.mark.parametrize("engine", ["duckdb", "polars"])
def test_child_process_runs_every_query_and_matches_the_oracle(tiny, tmp_path, engine):
    from krishiv_bench.memcap_child import main

    out_dir, _ = tiny
    for q in QUERIES:
        work = tmp_path / f"{engine}-{q}"
        assert main([engine, str(out_dir), q, "1g", str(work / "out"), str(work / "scratch")]) == 0
        got = memcap.checksum(work / "out", q)
        want = _checksum_sql(out_dir, q)
        assert compare_checksums(want, got) is None, (engine, q)


def _checksum_sql(data: Path, q: str) -> dict:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW sales AS SELECT * FROM read_parquet('{data}/sales/*.parquet')")
    con.execute(f"CREATE VIEW products AS SELECT * FROM read_parquet('{data}/products.parquet')")
    con.execute(f"CREATE VIEW result AS {SQL[q]}")
    cur = con.execute(f"SELECT {CHECKS[q]} FROM result")
    row = cur.fetchone()
    assert row is not None
    return dict(zip([d[0] for d in cur.description], row, strict=True))


def test_child_rejects_unknown_engine_and_bad_memory(tiny, tmp_path):
    from krishiv_bench.memcap_child import main, size_mib

    out_dir, _ = tiny
    with pytest.raises(SystemExit, match="unknown memcap engine 'flink'"):
        main(["flink", str(out_dir), "q1_filter_agg", "1g", str(tmp_path / "o"), str(tmp_path / "s")])
    assert size_mib("3g") == 3072
    with pytest.raises(ValueError, match="memory size must"):
        size_mib("3x")


# ------------------------------------------------------------------ command line
def _cli(monkeypatch, argv: list[str]) -> list[tuple]:
    from krishiv_bench import cli

    calls: list[tuple] = []
    monkeypatch.setattr(memcap, "run_memcap", lambda *a, **k: calls.append(a))
    cli.main(argv)
    return calls


def test_cli_run_memcap(monkeypatch):
    calls = _cli(
        monkeypatch, ["run", "--engine", "spark-local", "--suite", "memcap", "--rows", "50m", "--queries", "q4"]
    )
    assert calls == [("50m", "4.5g", ["spark"], "3g", 3, ["q4_highcard_agg"], 600.0)]
    calls = _cli(
        monkeypatch, ["run", "--engine", "krishiv", "--suite", "memcap", "--engine-memory", "1g", "--iterations", "5"]
    )
    assert calls == [("10m", "4.5g", ["krishiv"], "1g", 5, None, 600.0)]


def test_cli_reproduce_memcap_defaults_to_the_four_engines(monkeypatch):
    calls = _cli(monkeypatch, ["reproduce", "--suite", "memcap", "--rows", "150m", "--cap", "4.5g", "--reps", "2"])
    assert calls == [("150m", "4.5g", ["krishiv", "duckdb", "polars", "spark"], "3g", 2, None, 600.0)]
    calls = _cli(
        monkeypatch, ["reproduce", "--suite", "memcap", "--engines", "krishiv,duckdb", "--engine-memory", "1g"]
    )
    assert calls[0][2:4] == (["krishiv", "duckdb"], "1g")


@pytest.mark.parametrize(
    ("argv", "msg"),
    [
        (["run", "--engine", "datafusion", "--suite", "memcap"], "memcap engines are"),
        (["run", "--engine", "duckdb", "--suite", "memcap", "--queries", "q9"], "unknown memcap query"),
        (["run", "--engine", "duckdb", "--suite", "memcap", "--cap", "lots"], "memory size must"),
        (["run", "--engine", "spark", "--suite", "tpch"], "memcap engine"),
    ],
)
def test_cli_rejects_bad_memcap_arguments(monkeypatch, capsys, env, argv, msg):
    from krishiv_bench import cli

    monkeypatch.setattr(memcap, "run_one", lambda *a, **k: pytest.fail("no run may start"))
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    assert msg in capsys.readouterr().err


def test_cli_report_memcap(monkeypatch, env, capsys):
    from krishiv_bench import cli

    run_suite(env, Script({"duckdb": ["finished"]}), fake_check({"duckdb": {"n": 10}}), ["duckdb"], reps=1)
    cli.main(["report", "--suite", "memcap", "--rows", "10m"])
    assert "report_memcap_10m.md" in capsys.readouterr().out


def test_datasets_cli_memcap(monkeypatch, tmp_path, capsys):
    from krishiv_bench import datasets

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    seen = []
    monkeypatch.setattr(memcap_data, "generate", lambda label: seen.append(label))
    datasets.main(["memcap", "--rows", "10m", "50m"])
    assert seen == ["10m", "50m"]
    monkeypatch.setattr(memcap_data, "generate", lambda label: memcap_data.parse_rows(label))
    with pytest.raises(SystemExit):
        datasets.main(["memcap", "--rows", "7"])
    assert "whole number of millions" in capsys.readouterr().err


# ------------------------------------------------------------------ the real cgroup scope
def _scope_available() -> bool:
    import shutil
    import subprocess

    if shutil.which("systemd-run") is None:
        return False
    r = subprocess.run(
        ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=64M", "true"], capture_output=True, timeout=30
    )
    return r.returncode == 0


needs_scope = pytest.mark.skipif(not _scope_available(), reason="no systemd user scopes with a memory controller")


def _run_in_scope(code: str, cap: int):
    import subprocess
    import sys

    proc = subprocess.Popen(
        memcap.scope_command(cap, [sys.executable, "-c", code]),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    cg = memcap.find_cgroup(proc.pid)
    o = monitor(proc, cg, None, 60)
    out = proc.stdout.read() if proc.stdout else ""
    return o, out, cg


@needs_scope
def test_real_scope_classifies_a_process_that_exceeds_the_cap_as_killed():
    cap = 96 << 20
    # Allocate gradually: a single huge allocation can be killed between two monitor polls, before the scope's
    # peak is ever sampled near the cap.
    code = (
        "import time\nchunks = []\n"
        "for _ in range(150):\n"
        "    chunks.append(bytearray(4 << 20))\n"
        "    for i in range(0, 4 << 20, 4096): chunks[-1][i] = 1\n"
        "    time.sleep(0.005)\n"
    )
    o, _out, cg = _run_in_scope(code, cap)
    assert cg is not None
    assert o.returncode in (-9, 137, 1)
    assert classify(o, cap, got_result=False) == "killed"
    assert o.rss_peak_bytes <= cap


@needs_scope
def test_real_scope_classifies_a_clean_run_as_finished_and_reads_its_peak():
    cap = 512 << 20
    code = (
        "import time\nb = bytearray(40 << 20)\nfor i in range(0, len(b), 4096): b[i] = 1\n"
        "time.sleep(0.3)\nprint('RESULT {}')\n"
    )
    o, out, _cg = _run_in_scope(code, cap)
    assert classify(o, cap, got_result=memcap.read_result(out) is not None) == "finished"
    assert o.rss_peak_bytes >= 40 << 20


@needs_scope
def test_real_scope_timeout_kills_the_whole_tree():
    import subprocess
    import sys

    child = "import time; time.sleep(60)"
    code = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
    proc = subprocess.Popen(memcap.scope_command(CAP, [sys.executable, "-c", code]), stdout=subprocess.DEVNULL)
    cg = memcap.find_cgroup(proc.pid)
    o = monitor(proc, cg, None, 1.0)
    assert o.timed_out
    assert cg is not None
    try:  # the cgroup disappears with its last process: any read error means it is gone, which is the point
        remaining = (cg.path / "cgroup.procs").read_text().strip()
    except OSError:
        remaining = ""
    assert remaining == ""


@needs_scope
def test_run_one_runs_a_real_child_under_the_cap(tiny, tmp_path):
    out_dir, _ = tiny
    rec = memcap.run_one("duckdb", out_dir, "q4_highcard_agg", "1g", CAP, 120, tmp_path / "w")
    assert rec["status"] == "finished", rec
    assert rec["rows_out"] > 0
    assert rec["peak_rss_bytes"] > 0
    assert rec["wall_s"] >= rec["query_s"]
    failing = memcap.run_one("duckdb", tmp_path / "nodata", "q4_highcard_agg", "1g", CAP, 120, tmp_path / "w2")
    assert failing["status"] == "failed"
    assert "No files found" in failing["error"]


def test_cgroup_discovery_ignores_our_own_cgroup(monkeypatch):
    monkeypatch.setattr(memcap, "cgroup_of", lambda pid: "/user.slice/run-1.scope")
    assert memcap.find_cgroup(1, wait_s=0.05) is None  # a process that never leaves our scope is not "its" scope
    monkeypatch.setattr(memcap, "cgroup_of", lambda pid: None if pid != "self" else "/x.scope")
    assert memcap.find_cgroup(1, wait_s=0.05) is None
    paths = {"self": "/user.slice/run-1.scope", 7: "/user.slice/run-2.scope"}
    monkeypatch.setattr(memcap, "cgroup_of", lambda pid: paths[pid])
    cg = memcap.find_cgroup(7, wait_s=0.05)
    assert cg is not None
    assert cg.path == memcap.CGROUP_ROOT / "user.slice/run-2.scope"


def test_krishiv_child_drives_the_runner_with_the_memory_limit_and_spill_dir(tiny, tmp_path, monkeypatch, capsys):
    import stat
    import sys
    import textwrap

    from krishiv_bench.memcap_child import main

    argv_log = tmp_path / "argv.json"
    runner = tmp_path / "runner"
    runner.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, os, sys
            from urllib.parse import unquote
            import pyarrow as pa, pyarrow.parquet as pq
            json.dump(sys.argv[1:], open({str(argv_log)!r}, "w"))
            print("READY 9.9.9 sales,products", flush=True)
            for line in sys.stdin:
                cmd = line.split()
                if cmd[0] == "QUIT":
                    break
                out = unquote(cmd[2]); os.makedirs(out)
                pq.write_table(pa.table({{"a": [1, 2, 3]}}), out + "/part-0.parquet")
                print("OK 3 0.5", flush=True)
            """
        )
    )
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("KRISHIV_BENCH_RUNNER", str(runner))
    data, _ = tiny
    work = tmp_path / "w"
    assert main(["krishiv", str(data), "q1_filter_agg", "1g", str(work / "out"), str(work / "scratch")]) == 0
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RESULT "))
    assert (res["engine"], res["version"], res["rows_out"]) == ("krishiv", "9.9.9", 3)
    args = json.loads(argv_log.read_text())
    assert args[args.index("--memory-limit") + 1] == "1g"
    assert args[args.index("--threads") + 1] == "8"
    assert f"krishiv.spill.dir={work / 'scratch'}" in args
    assert (work / "scratch").stat().st_mode & 0o777 == 0o700


def test_spark_child_sets_driver_memory_and_scratch(tiny, tmp_path, monkeypatch, capsys):
    import sys
    import types

    from krishiv_bench.memcap_child import main

    conf: dict[str, str] = {}
    written: list[str] = []

    class Frame:
        def createOrReplaceTempView(self, name):  # pyspark's name
            conf[f"view:{name}"] = "1"

        @property
        def write(self):
            return self

        def mode(self, _m):
            return self

        def parquet(self, path):
            written.append(path)
            Path(path).mkdir(parents=True)
            pq.write_table(pa.table({"a": [1]}), Path(path) / "p.parquet")

    class Builder:
        def master(self, m):
            conf["master"] = m
            return self

        def config(self, k, v):
            conf[k] = v
            return self

        def getOrCreate(self):
            return types.SimpleNamespace(
                sparkContext=types.SimpleNamespace(setLogLevel=lambda _l: None),
                read=types.SimpleNamespace(parquet=lambda _p: Frame()),
                sql=lambda _q: Frame(),
                stop=lambda: conf.__setitem__("stopped", "1"),
            )

    pyspark = types.ModuleType("pyspark")
    setattr(pyspark, "__version__", "4.2.0")  # noqa: B010
    sql = types.ModuleType("pyspark.sql")
    setattr(sql, "SparkSession", types.SimpleNamespace(builder=Builder()))  # noqa: B010
    monkeypatch.setitem(sys.modules, "pyspark", pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    data, _ = tiny
    work = tmp_path / "w"
    assert main(["spark", str(data), "q3_window_latest", "3g", str(work / "out"), str(work / "scratch")]) == 0
    assert conf["spark.driver.memory"] == "3072m"
    assert conf["spark.local.dir"] == str(work / "scratch")
    assert conf["master"] == "local[8]"
    assert conf["stopped"] == "1"  # the session is stopped on the way out
    assert written == [str(work / "out" / "spark")]
    assert '"version": "4.2.0"' in capsys.readouterr().out


def test_timestamp_literal_dialect():
    assert sql_for("duckdb", "q1_filter_agg") == SQL["q1_filter_agg"]
    assert "CAST('2026-06-01 00:00:00' AS TIMESTAMP_NTZ)" in sql_for("krishiv", "q1_filter_agg")
    assert "TIMESTAMP '" not in sql_for("krishiv", "q1_filter_agg")
    for q in QUERIES[1:]:
        assert sql_for("krishiv", q) == SQL[q]


def test_monitor_keeps_the_limit_hits_seen_before_the_scope_vanishes():
    clock = Clock()
    proc = FakeProc(2, -9)
    cg = FakeCgroup([10, 20, None], [0, 0, None], [5, 6, None], hits=[0, 3, None])
    o = monitor(proc, cg, None, None, clock=clock, sleep=clock.sleep)
    assert o.limit_hits == 3
    assert classify(o, CAP, got_result=False) == "killed"  # the peak was never sampled near the cap
