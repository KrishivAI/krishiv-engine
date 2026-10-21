"""ClickBench query fetching: pinned download, SHA-256 check, cache and run-time adaptations (synthetic text only)."""

from __future__ import annotations

import hashlib
import urllib.error
from email.message import Message

import pytest

from krishiv_bench import clickbench_queries as cq
from krishiv_bench.runner import load_queries

COMMIT = "c" * 40


def _file(lines: list[str]) -> tuple[bytes, dict[str, tuple[str, str]]]:
    data = ("\n".join(lines) + "\n").encode()
    sha = hashlib.sha256(data).hexdigest()
    return data, {"duckdb-parquet": ("a/q.sql", sha), "clickhouse": ("b/q.sql", sha)}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path / "data"))
    return tmp_path / "data"


def test_engine_picks_its_upstream_file():
    assert cq.source_name("clickhouse") == "clickhouse"
    assert cq.source_name("duckdb") == "duckdb-parquet"
    assert cq.source_name("polars") == "duckdb-parquet"


def test_fetch_verifies_hash_and_caches(data_dir):
    data, pins = _file(["SELECT 1;"])
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return data

    assert cq.fetch_file("clickhouse", fetch, pins, COMMIT) == data.decode()
    assert calls == [f"https://raw.githubusercontent.com/ClickHouse/ClickBench/{COMMIT}/b/q.sql"]
    assert (data_dir / "clickbench" / "queries" / COMMIT[:12] / "clickhouse.sql").read_bytes() == data
    assert cq.fetch_file("clickhouse", fetch, pins, COMMIT) == data.decode()
    assert len(calls) == 1  # second call served from the cache


def test_fetch_rejects_content_that_does_not_match_the_pin(data_dir):
    _, pins = _file(["SELECT 1;"])
    with pytest.raises(ValueError, match="does not match the pinned"):
        cq.fetch_file("clickhouse", lambda url: b"SELECT 2;\n", pins, COMMIT)
    assert not (data_dir / "clickbench" / "queries").exists()  # nothing unverified is cached


def test_corrupt_cache_entry_is_fetched_again(data_dir):
    data, pins = _file(["SELECT 1;"])
    cached = data_dir / "clickbench" / "queries" / COMMIT[:12] / "clickhouse.sql"
    cached.parent.mkdir(parents=True)
    cached.write_text("tampered")
    assert cq.fetch_file("clickhouse", lambda url: data, pins, COMMIT) == data.decode()
    assert cached.read_bytes() == data


def test_load_checks_query_count_and_numbers_queries():
    data, pins = _file(["SELECT 1;", "SELECT 2;"])
    got = cq.load("polars", lambda url: data, pins, COMMIT, expected=2)
    assert got == {"q01": "SELECT 1;", "q02": "SELECT 2;"}
    with pytest.raises(ValueError, match="has 2 queries, expected 3"):
        cq.load("polars", lambda url: data, pins, COMMIT, expected=3)


def test_adaptations_rewrite_dates_lengths_and_time_functions():
    sql = (
        "SELECT STRLEN(x), extract(minute FROM toDateTime(EventTime)), DATE_TRUNC('minute', toDateTime(EventTime)) "
        "FROM t WHERE EventDate >= '1970-01-02' AND EventDate <= '2013-07-31';"
    )
    assert cq.adapt(sql, "q99") == (
        "SELECT length(x), CAST(FLOOR(EventTime % 3600 / 60) AS BIGINT), EventTime - EventTime % 60 "
        "FROM t WHERE EventDate >= 1 AND EventDate <= 15917;"
    )
    assert cq.adapt("SELECT extract(minute FROM EventTime) FROM t;", "q99") == (
        "SELECT CAST(FLOOR(EventTime % 3600 / 60) AS BIGINT) FROM t;"
    )


def test_invalid_date_literal_is_rejected():
    with pytest.raises(ValueError, match="invalid date literal"):
        cq.adapt("SELECT 1 FROM t WHERE EventDate >= '2013-13-45';", "q99")


def test_tie_break_extends_or_adds_order_by_before_limit():
    rules = {"q1": ["k"], "q2": ["a", "b"], "q3": ["k"]}
    assert cq.adapt("SELECT k FROM t ORDER BY c DESC LIMIT 5;", "q1", rules) == (
        "SELECT k FROM t ORDER BY c DESC, k LIMIT 5;"
    )
    assert cq.adapt("SELECT a, b FROM t GROUP BY a, b LIMIT 5 OFFSET 2;", "q2", rules) == (
        "SELECT a, b FROM t GROUP BY a, b ORDER BY a, b LIMIT 5 OFFSET 2;"
    )
    assert (
        cq.adapt("SELECT k FROM t ORDER BY c DESC LIMIT 5;", "q9", rules) == "SELECT k FROM t ORDER BY c DESC LIMIT 5;"
    )
    with pytest.raises(ValueError, match="without LIMIT"):
        cq.adapt("SELECT k FROM t ORDER BY c;", "q3", rules)


def test_production_tie_breaks_cover_q12_and_q24():
    q12 = (
        "SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' "
        "GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10;"
    )
    q24 = "SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;"
    assert cq.adapt(q12, "q12").endswith("ORDER BY u DESC, MobilePhone, MobilePhoneModel LIMIT 10;")
    assert cq.adapt(q24, "q24").endswith("ORDER BY EventTime, WatchID, ClientIP LIMIT 10;")


def test_pins_name_both_upstream_files_with_sha256():
    assert set(cq.PINS) == {"duckdb-parquet", "clickhouse"}
    assert cq.PINS == {
        "duckdb-parquet": (
            "duckdb-parquet/queries.sql",
            "e84ea26e1ed13a1dc1e46e76ddc9e49ead67a0444cdd5ae92085daad269c5508",
        ),
        "clickhouse": (
            "clickhouse/queries.sql",
            "a7d6673357348ee9680443216b6f26f30d1dce9f313b419d38502417b2c2a219",
        ),
    }
    assert cq.COMMIT == "0e10b8434eef001b43156b860a4e2e3665a35d51"


def test_load_queries_uses_the_fetcher_for_clickbench(monkeypatch):
    seen: list[str] = []

    def fake_load(engine: str) -> dict[str, str]:
        seen.append(engine)
        return {"q01": "SELECT 1;", "q02": "SELECT 2;"}

    monkeypatch.setattr(cq, "load", fake_load)
    qs = load_queries("clickbench", "clickhouse", ["q02"])
    assert [(q.id, q.sql, q.suite) for q in qs] == [("q02", "SELECT 2;", "clickbench")]
    assert seen == ["clickhouse"]


class _Resp:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.body


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", Message(), None)


def test_http_get_retries_transient_errors_only(monkeypatch):
    monkeypatch.setattr(cq.time, "sleep", lambda s: None)
    outcomes: list = [_http_error(503), urllib.error.URLError("reset"), _Resp(b"ok")]
    monkeypatch.setattr(cq.urllib.request, "urlopen", lambda url, timeout: _pop(outcomes))
    assert cq._http_get("https://x") == b"ok"
    assert outcomes == []


def test_http_get_does_not_retry_a_client_error(monkeypatch):
    monkeypatch.setattr(cq.time, "sleep", lambda s: None)
    outcomes: list = [_http_error(404), _Resp(b"never")]
    monkeypatch.setattr(cq.urllib.request, "urlopen", lambda url, timeout: _pop(outcomes))
    with pytest.raises(urllib.error.HTTPError) as e:
        cq._http_get("https://x")
    assert e.value.code == 404
    assert len(outcomes) == 1


def test_http_get_gives_up_after_the_attempts(monkeypatch):
    monkeypatch.setattr(cq.time, "sleep", lambda s: None)
    outcomes: list = [_http_error(500), _http_error(502), _http_error(503), _Resp(b"late")]
    monkeypatch.setattr(cq.urllib.request, "urlopen", lambda url, timeout: _pop(outcomes))
    with pytest.raises(urllib.error.HTTPError) as e:
        cq._http_get("https://x", attempts=3)
    assert e.value.code == 503
    assert len(outcomes) == 1


def _pop(outcomes: list):
    o = outcomes.pop(0)
    if isinstance(o, Exception):
        raise o
    return o
