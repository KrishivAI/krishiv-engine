"""ClickHouse adapter (chdb): option parsing, session setup against a fake chdb, a real smoke test, dialect verdicts."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pytest

from krishiv_bench import compare, reproduce
from krishiv_bench.adapters import Dataset, Engine, Query, make_engine
from krishiv_bench.adapters.clickhouse import ClickHouseEngine, parse_memory_limit, sql_string
from krishiv_bench.dialect import dialect_reason, unsupported_reason
from krishiv_bench.paths import queries_dir
from krishiv_bench.runner import load_queries


@pytest.mark.parametrize(
    ("text", "expected"),
    [("12g", 12 << 30), ("12GiB", 12 << 30), ("12GB", 12 << 30), ("1024", 1024), ("3 m", 3 << 20), (7, 7)],
)
def test_parse_memory_limit_accepts(text, expected):
    assert parse_memory_limit(text) == expected


@pytest.mark.parametrize("text", ["0", "-1", "012g", "abc", "1.5g", "", "12gg", "g", "0g"])
def test_parse_memory_limit_rejects(text):
    with pytest.raises(ValueError, match="invalid memory limit"):
        parse_memory_limit(text)


def test_sql_string_escapes_quote_and_backslash():
    assert sql_string(r"/d/it's\a") == r"'/d/it\'s\\a'"


def test_rejects_non_positive_threads_and_bad_memory():
    with pytest.raises(ValueError, match="invalid thread count"):
        ClickHouseEngine(threads=-2)
    with pytest.raises(ValueError, match="invalid memory limit"):
        ClickHouseEngine(threads=1, memory_limit="lots")


def test_registered_in_the_adapter_registry():
    eng = make_engine("clickhouse", threads=2, memory_limit="1g")
    assert isinstance(eng, ClickHouseEngine)
    assert isinstance(eng, Engine)
    cfg = eng.config()
    assert cfg["threads"] == 2
    assert cfg["max_memory_usage"] == 1 << 30
    assert cfg["spill_threshold_bytes"] == 1 << 29
    assert cfg["join_use_nulls"] is True


class _Result:
    def __init__(self, payload):
        self._payload = payload

    def bytes(self):
        return self._payload


class FakeSession:
    instances: ClassVar[list[FakeSession]] = []
    fail_on: ClassVar[str | None] = None

    def __init__(self, path: str):
        self.path = path
        self.statements: list[str] = []
        self.closed = False
        FakeSession.instances.append(self)

    def query(self, sql: str, fmt: str = "CSV"):
        self.statements.append(sql)
        if FakeSession.fail_on and FakeSession.fail_on in sql:
            raise RuntimeError("Code: 60. DB::Exception: boom")
        if fmt == "ArrowTable":
            return pa.table({"a": [1, 2, 3], "b": ["x", None, "z"]})
        return _Result(b'"26.9.2.1"\n')

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_chdb(monkeypatch, tmp_path):
    FakeSession.instances.clear()
    FakeSession.fail_on = None
    pkg = types.ModuleType("chdb")
    setattr(pkg, "__version__", "9.9.9")  # noqa: B010
    sess = types.ModuleType("chdb.session")
    setattr(sess, "Session", FakeSession)  # noqa: B010
    setattr(pkg, "session", sess)  # noqa: B010
    monkeypatch.setitem(sys.modules, "chdb", pkg)
    monkeypatch.setitem(sys.modules, "chdb.session", sess)
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    return FakeSession


def _dataset(root: Path, tables: list[str]) -> Dataset:
    for t in tables:
        (root / f"{t}.parquet").write_bytes(b"")
    return Dataset("fake", 1, root, tables)


def test_setup_pins_threads_memory_and_registers_views(fake_chdb, tmp_path):
    root = tmp_path / "it's"
    root.mkdir()
    escaped_root = str(root).replace("'", "\\'")
    eng = ClickHouseEngine(threads=3, memory_limit="2g")
    eng.setup(_dataset(root, ["t1", "t2"]))
    (sess,) = fake_chdb.instances
    assert sess.path == str(tmp_path / "data" / "tmp" / f"clickhouse-{os.getpid()}")
    assert Path(sess.path).is_dir()
    assert sess.statements[:6] == [
        "SET max_threads = 3",
        "SET join_use_nulls = 1",
        f"SET max_memory_usage = {2 << 30}",
        f"SET max_bytes_before_external_group_by = {1 << 30}",
        f"SET max_bytes_before_external_sort = {1 << 30}",
        f"CREATE VIEW t1 AS SELECT * FROM file('{escaped_root}/t1.parquet', Parquet)",
    ]
    assert sess.statements[6].startswith("CREATE VIEW t2 AS")
    assert eng.version() == "chdb 9.9.9 (ClickHouse 26.9.2.1)"
    eng.teardown()
    assert sess.closed
    assert not Path(sess.path).exists()


def test_run_and_timed_run_return_rows_and_count(fake_chdb, tmp_path):
    eng = ClickHouseEngine(threads=1, memory_limit="1g")
    eng.setup(_dataset(tmp_path, ["t"]))
    q = Query("q1", "SELECT a, b FROM t", "s")
    assert eng.run(q) == [(1, "x"), (2, None), (3, "z")]
    n, secs = eng.timed_run(q)
    assert n == 3
    assert secs >= 0
    assert fake_chdb.instances[0].statements[-1] == "SELECT a, b FROM t"
    eng.teardown()


def test_run_before_setup_is_an_error():
    with pytest.raises(AssertionError, match="setup"):
        ClickHouseEngine(threads=1).run(Query("q", "SELECT 1", "s"))


def test_scratch_and_session_cleaned_when_setup_fails(fake_chdb, tmp_path):
    fake_chdb.fail_on = "max_memory_usage"
    eng = ClickHouseEngine(threads=1, memory_limit="1g")
    with pytest.raises(RuntimeError, match="boom"):
        eng.setup(_dataset(tmp_path, ["t"]))
    (sess,) = fake_chdb.instances
    assert sess.closed
    assert not Path(sess.path).exists()


def test_missing_parquet_fails_setup_and_cleans_up(fake_chdb, tmp_path):
    eng = ClickHouseEngine(threads=1, memory_limit="1g")
    with pytest.raises(FileNotFoundError, match="no Parquet source"):
        eng.setup(Dataset("fake", 1, tmp_path, ["absent"]))
    assert not Path(fake_chdb.instances[0].path).exists()


# ---------------------------------------------------------------- real engine (skipped when chdb is not installed)
@pytest.fixture
def tiny_decimal(tmp_path_factory) -> Dataset:
    import duckdb

    root = tmp_path_factory.mktemp("chtiny")
    duckdb.sql(
        "COPY (SELECT range AS a, range % 3 AS g, CAST(range AS DECIMAL(12, 2)) / 4 AS d, "
        "CASE WHEN range % 5 = 0 THEN NULL ELSE 'v' || range END AS s, "
        "DATE '2024-01-01' + CAST(range AS INTEGER) AS dt "
        f"FROM range(100)) TO '{root / 't.parquet'}' (FORMAT parquet)"
    )
    return Dataset("tiny", 1, root, ["t"])


def test_real_chdb_matches_duckdb_on_types_and_nulls(tiny_decimal, tmp_path, monkeypatch):
    pytest.importorskip("chdb", reason="chdb (embedded ClickHouse) is not installed; the smoke test needs it")
    import duckdb

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    eng = ClickHouseEngine(threads=2, memory_limit="1g")
    eng.setup(tiny_decimal)
    try:
        assert "ClickHouse " in eng.version()
        sql = "SELECT g, count(*), sum(d), count(s), min(dt) FROM t GROUP BY g ORDER BY g"
        got = eng.run(Query("q", sql, "s"))
        want = duckdb.sql(sql.replace("FROM t", f"FROM read_parquet('{tiny_decimal.root / 't.parquet'}')")).fetchall()
        assert compare.rows_match(got, want, ordered=True) == (True, "ok")
        n, secs = eng.timed_run(Query("q", sql, "s"))
        assert n == 3
        assert secs > 0
        with pytest.raises(Exception, match="UNKNOWN_IDENTIFIER"):
            eng.run(Query("bad", "SELECT nope FROM t", "s"))
        assert eng.run(Query("again", "SELECT 1", "s")) == [(1,)]  # usable after an engine error
    finally:
        eng.teardown()
    assert not (tmp_path / "data" / "tmp" / f"clickhouse-{os.getpid()}").exists()


# ---------------------------------------------------------------- verdicts and query set
class _Rows(Engine):
    def __init__(self, rows):
        self.rows = rows

    def version(self) -> str:
        return "fake"

    def setup(self, dataset: Dataset) -> None:
        pass

    def run(self, query: Query):
        return self.rows


def _q(suite: str, qid: str) -> Query:
    return Query(qid, "SELECT x FROM t ORDER BY x", suite)


def test_known_dialect_difference_is_recorded_with_its_reason():
    qm = {n: _q("tpch", "q08") for n in ("duckdb", "clickhouse")}
    verdict, _ = reproduce.correctness(qm, {"duckdb": _Rows([(0.0414855,)]), "clickhouse": _Rows([(0.0414,)])})
    v = verdict["clickhouse"]
    assert v["status"] == "mismatch-dialect"
    assert "truncates" in v["reason"]
    assert "value mismatch at row 0" in v["reason"]


def test_unlisted_difference_stays_a_plain_mismatch():
    qm = {n: _q("tpch", "q01") for n in ("duckdb", "clickhouse")}
    verdict, _ = reproduce.correctness(qm, {"duckdb": _Rows([(1,)]), "clickhouse": _Rows([(2,)])})
    assert verdict["clickhouse"]["status"] == "mismatch"


def test_listed_query_that_matches_is_ok_and_other_engines_are_not_affected():
    qm = {n: _q("tpch", "q08") for n in ("duckdb", "clickhouse", "polars")}
    engines = {"duckdb": _Rows([(1,)]), "clickhouse": _Rows([(1,)]), "polars": _Rows([(2,)])}
    verdict, _ = reproduce.correctness(qm, engines)
    assert verdict["clickhouse"]["status"] == "ok"
    assert verdict["polars"]["status"] == "mismatch"
    assert dialect_reason("polars", "tpch", "q08") is None


def test_tpcds_is_skipped_with_a_stated_reason():
    assert unsupported_reason("clickhouse", "tpcds") is not None
    assert unsupported_reason("duckdb", "tpcds") is None
    with pytest.raises(SystemExit, match="TPC-DS is not run for ClickHouse"):
        load_queries("tpcds", "clickhouse")


def test_tpch_queries_carry_provenance_and_are_one_statement():
    files = sorted((queries_dir("tpch") / "clickhouse").glob("q*.sql"))
    assert [f.stem for f in files] == [f"q{i:02d}" for i in range(1, 23)]
    for f in files:
        text = f.read_text()
        lines = text.splitlines()
        assert lines[0].startswith("-- Source: ClickHouse/ClickHouse tests/benchmarks/tpc-h/queries/query_"), f.name
        assert lines[0].endswith("(Apache-2.0)."), f.name
        assert lines[1].startswith("-- Adaptations: "), f.name
        # ORDER BY detection strips quoted text before comments, so a quote in a comment would hide it
        assert all("'" not in ln for ln in lines if ln.startswith("--")), f.name
        assert "\n".join(ln for ln in lines if not ln.startswith("--")).count(";") == 1, f.name


def test_engine_override_is_picked_up_by_the_loader():
    ch = {q.id: q.sql for q in load_queries("tpch", "clickhouse")}
    shared = {q.id: q.sql for q in load_queries("tpch", "duckdb")}
    assert ch.keys() == shared.keys()
    assert ch["q01"].startswith("-- Source: ClickHouse/ClickHouse")
    assert not shared["q01"].startswith("--")
    assert {q.id for q in load_queries("clickbench", "clickhouse")} == {
        q.id for q in load_queries("clickbench", "duckdb")
    }


def test_worker_process_runs_clickhouse_and_reports_engine_errors(tiny_decimal):
    pytest.importorskip("chdb", reason="chdb (embedded ClickHouse) is not installed; the worker smoke test needs it")
    from krishiv_bench.adapters.process import EngineError, ProcessEngine

    eng = ProcessEngine("clickhouse", 2, 1, 120)
    eng.setup(tiny_decimal)
    try:
        assert eng.run(Query("q", "SELECT count(*), max(a) FROM t", "s")) == [(100, 99)]
        assert eng.version().startswith("chdb ")
        assert eng.config()["threads"] == 2
        with pytest.raises(EngineError, match="UNKNOWN_IDENTIFIER"):
            eng.run(Query("bad", "SELECT nope FROM t", "s"))
        assert eng.run(Query("again", "SELECT 1", "s")) == [(1,)]
    finally:
        eng.teardown()


def test_version_before_setup_names_only_the_library(fake_chdb):
    assert ClickHouseEngine(threads=1).version() == "chdb 9.9.9"


def test_measure_brackets_report_resource_use(fake_chdb, tmp_path):
    eng = ClickHouseEngine(threads=1, memory_limit="1g")
    eng.setup(_dataset(tmp_path, ["t"]))
    try:
        eng.begin_measure()
        m = eng.end_measure()
        assert m.peak_rss_bytes is not None
        assert m.peak_rss_bytes > 0
        assert m.baseline_rss_bytes is not None
        assert m.cpu_seconds is not None
        assert m.cpu_seconds >= 0
    finally:
        eng.teardown()
