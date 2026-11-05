"""The reference cache: DuckDB's answers and timings stored once and reused (reproduce, abba)."""

from __future__ import annotations

import datetime as dt
import decimal
import json
from pathlib import Path
from typing import Any

import pytest

from krishiv_bench import abba, reproduce
from krishiv_bench.adapters import Dataset, Engine, Query
from krishiv_bench.refcache import FORMAT, MAX_STORED_ROWS, RefCache, dataset_signature


@pytest.fixture
def ds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dataset:
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    root = tmp_path / "tpch" / "sf1"
    root.mkdir(parents=True)
    (root / "t.parquet").write_bytes(b"abc")
    return Dataset("tpch", 1, root, ["t"])


def cache(tmp_path: Path, **kw) -> RefCache:
    return RefCache(root=tmp_path / "rc", engine="duckdb", version="1.5.6", **kw)


Q = Query("q01", "SELECT 1", "tpch")


def test_answer_roundtrips_typed_rows(tmp_path: Path, ds: Dataset):
    c = cache(tmp_path)
    key = c.answer_key("tpch", ds, Q)
    rows = [
        (1, "a", decimal.Decimal("1.50"), dt.date(2024, 1, 2), None, dt.datetime(2024, 1, 2, 3, 4, 5, 6)),
        (2, "b", decimal.Decimal("0"), None, 1.5, b"\x00\xff"),
    ]
    assert c.load(key) is None
    assert c.store(key, rows, {"note": "x"})
    hit = c.load(key)
    assert hit is not None
    entry, got = hit
    assert got == rows
    assert entry["note"] == "x"
    assert entry["fields"] == key
    assert "created_utc" in entry


def test_key_changes_with_every_identity_field(tmp_path: Path, ds: Dataset):
    c = cache(tmp_path)
    base = c.answer_key("tpch", ds, Q)
    c.store(base, [(1,)])
    for other in (
        c.answer_key("tpcds", ds, Q),  # suite
        c.answer_key("tpch", Dataset("tpch", 10, ds.root, ["t"]), Q),  # scale factor
        c.answer_key("tpch", ds, Query("q02", Q.sql, "tpch")),  # query id
        c.answer_key("tpch", ds, Query("q01", "SELECT 2", "tpch")),  # query text
        c.answer_key("tpch", ds, Query("q01", Q.sql, "tpch", check_sql="SELECT count(*)")),  # checksum text
        RefCache(root=c.root, engine="duckdb", version="1.6.0").answer_key("tpch", ds, Q),  # engine version
        c.timing_key("tpch", ds, Q, 12, 32, 6),  # kind
    ):
        assert c.load(other) is None
    (ds.root / "t.parquet").write_bytes(b"abcd")  # the data changed
    assert c.load(c.answer_key("tpch", ds, Q)) is None


def test_timing_key_includes_threads_memory_and_samples(tmp_path: Path, ds: Dataset):
    c = cache(tmp_path)
    k = c.timing_key("tpch", ds, Q, 12, 32, 6)
    c.store(k, [(1,)])
    assert c.load(c.timing_key("tpch", ds, Q, 12, 32, 6)) is not None
    for other in ((8, 32, 6), (12, 16, 6), (12, 32, 4)):
        assert c.load(c.timing_key("tpch", ds, Q, *other)) is None


def test_refresh_ignores_what_is_stored_and_overwrites_it(tmp_path: Path, ds: Dataset):
    key = cache(tmp_path).answer_key("tpch", ds, Q)
    cache(tmp_path).store(key, [(1,)])
    fresh = cache(tmp_path, refresh=True)
    assert fresh.load(key) is None
    fresh.store(key, [(2,)])
    hit = cache(tmp_path).load(key)
    assert hit is not None
    assert hit[1] == [(2,)]


def test_a_huge_answer_is_not_stored(tmp_path: Path, ds: Dataset):
    c = cache(tmp_path)
    key = c.answer_key("tpch", ds, Q)
    assert not c.store(key, [(0,)] * (MAX_STORED_ROWS + 1))
    assert c.load(key) is None
    assert not list(c.root.glob("*")) if c.root.exists() else True


def test_a_damaged_or_foreign_entry_is_a_miss(tmp_path: Path, ds: Dataset):
    c = cache(tmp_path)
    key = c.answer_key("tpch", ds, Q)
    c.store(key, [(1,)])
    meta, rows = c._paths(key)
    entry = json.loads(meta.read_text())
    entry["fields"]["version"] = "other"  # a digest collision or an edited file
    meta.write_text(json.dumps(entry))
    assert c.load(key) is None
    meta.write_text("{not json")
    assert c.load(key) is None
    c.store(key, [(1,)])
    rows.write_bytes(b"not gzip")
    assert c.load(key) is None


def test_format_is_part_of_the_key(tmp_path: Path, ds: Dataset):
    assert cache(tmp_path).answer_key("tpch", ds, Q)["format"] == FORMAT


def test_dataset_signature_prefers_the_manifest(tmp_path: Path, ds: Dataset):
    by_files = dataset_signature(ds)
    assert by_files.startswith("files:")
    man = tmp_path / "data" / "manifests"
    man.mkdir(parents=True)
    (man / "tpch-sf1.sha256").write_text("deadbeef")
    sig = dataset_signature(ds)
    assert sig.startswith("manifest:")
    assert sig != by_files
    (man / "tpch-sf1.sha256").write_text("cafe")
    assert dataset_signature(ds) != sig


class Counting(Engine):
    def __init__(self, rows):
        self.rows, self.runs = rows, 0

    def version(self) -> str:
        return "fake"

    def setup(self, dataset) -> None:
        pass

    def run(self, query):
        self.runs += 1
        return self.rows


def test_correctness_reads_duckdbs_answer_from_the_cache_and_still_checks_the_others(tmp_path: Path, ds: Dataset):
    c = cache(tmp_path)
    qm = {n: Query("q01", "SELECT x FROM t ORDER BY x", "tpch") for n in ("duckdb", "krishiv")}

    def run(krishiv_rows, cache_):
        duck, kri = Counting([(1,), (2,)]), Counting(krishiv_rows)
        verdict, n = reproduce.correctness(qm, {"duckdb": duck, "krishiv": kri}, cache_, "tpch", ds)
        return verdict, n, duck.runs, kri.runs

    verdict, n, duck_runs, _ = run([(1,), (2,)], c)
    assert (n, duck_runs, verdict["krishiv"]["status"]) == (2, 1, "ok")
    verdict, n, duck_runs, kri_runs = run([(1,), (3,)], c)  # second run: DuckDB is not run, Krishiv still is
    assert (n, duck_runs, kri_runs) == (2, 0, 1)
    assert verdict["krishiv"]["status"] == "mismatch"
    _, _, duck_runs, _ = run([(1,), (2,)], RefCache(root=c.root, engine="duckdb", version="1.5.6", refresh=True))
    assert duck_runs == 1  # --refresh-reference
    _, _, duck_runs, _ = run([(1,), (2,)], None)
    assert duck_runs == 1  # --no-reference-cache


def test_abba_uses_the_cached_answer_but_times_duckdb_live(tmp_path: Path, ds: Dataset, monkeypatch):
    c = cache(tmp_path)
    key = c.answer_key("tpch", ds, Q)
    c.store(key, [(1,)])
    duck = Counting([(9,)])
    ran = []

    class Krishiv:
        timeout_s = None

        def run(self, q):
            return [(1,)]

        def timed_run(self, q):
            ran.append("k")
            return 1, 0.01

    krishiv: Any = Krishiv()
    monkeypatch.setattr(abba, "_duck_timed", lambda d, q, t: ran.append("d") or 0.02)
    entry = abba._one_query(Q, krishiv, duck, 1, 10, None, c, "tpch", ds)
    assert duck.runs == 0  # its answer came from the cache
    assert entry["rows"] == 1
    assert "reference_cached_utc" in entry
    assert ran == ["k", "d", "d", "k"]  # the ABBA timing is live
    assert abba._one_query(Q, krishiv, duck, 1, 10, None, None, "tpch", ds)["rows"] == 1
    assert duck.runs == 1  # without a cache DuckDB answers itself


def test_sequential_reference_is_measured_once_then_replayed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import duckdb

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    root = tmp_path / "tiny"
    root.mkdir()
    duckdb.sql(f"COPY (SELECT range AS a, range % 3 AS g FROM range(1000)) TO '{root / 't.parquet'}' (FORMAT parquet)")
    tiny = Dataset("tiny", 1, root, ["t"])
    qm = {"duckdb": Query("q01", "SELECT g, count(*) FROM t GROUP BY g ORDER BY g", "tpch")}
    c = cache(tmp_path)

    first, infos = reproduce.sequential_query(qm, ["duckdb"], tiny, 2, 1, None, 2, 60, False, c, "tiny")
    assert first["engines"]["duckdb"]["status"] == "ok"
    assert "cached_utc" not in first["engines"]["duckdb"]
    assert first["rows"] == 3

    def no_worker(*a, **k):
        raise AssertionError("the reference engine must not be started on a cache hit")

    monkeypatch.setattr(reproduce, "ProcessEngine", no_worker)
    second, infos2 = reproduce.sequential_query(qm, ["duckdb"], tiny, 2, 1, None, 2, 60, False, c, "tiny")
    d1, d2 = first["engines"]["duckdb"], second["engines"]["duckdb"]
    assert "cached_utc" in d2
    assert d2["samples_s"] == d1["samples_s"]
    assert d2["median_s"] == d1["median_s"]
    assert d2["result_check"] == d1["result_check"]
    assert second["rows"] == 3
    assert infos2["duckdb"] == infos["duckdb"]
    # another thread count is another measurement: the engine runs (and here the stub refuses)
    with pytest.raises(AssertionError, match="must not be started"):
        reproduce.sequential_query(qm, ["duckdb"], tiny, 4, 1, None, 2, 60, False, c, "tiny")


def test_cold_reproduce_and_abba_evict_before_every_timed_run(tmp_path: Path, ds: Dataset, monkeypatch):
    evicted: list[Path] = []
    monkeypatch.setattr(reproduce, "evict_page_cache", lambda root: evicted.append(root))
    monkeypatch.setattr(abba, "evict_page_cache", lambda root: evicted.append(root))

    class Timed(Counting):
        def timed_run(self, query):
            return 1, 0.01

        def result_checksum(self):
            return None

    reproduce._timed(Timed([]), Q, 1, None, None)
    assert evicted == []  # warm
    reproduce._timed(Timed([]), Q, 1, None, ds.root)
    assert evicted == [ds.root]

    class Krishiv:
        def run(self, q):
            return [(1,)]

        def timed_run(self, q):
            return 1, 0.01

    krishiv: Any = Krishiv()
    monkeypatch.setattr(abba, "_duck_timed", lambda d, q, t: 0.02)
    evicted.clear()
    abba._one_query(Q, krishiv, Counting([(1,)]), 2, 10, None, None, "tpch", ds, ds.root)
    assert evicted == [ds.root] * 8  # two blocks of A B B A
