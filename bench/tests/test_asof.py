"""asof suite: generator determinism and ties, digests, query loading, and the queries against a DuckDB oracle."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from krishiv_bench import asof_data
from krishiv_bench.runner import load_queries

TINY = {"trades": 3_000, "quotes": 12_000}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return asof_data.generate(1, tmp_path_factory.mktemp("asof") / "tiny", rows=TINY)


def test_generator_is_deterministic(tiny, tmp_path):
    again = asof_data.generate(1, tmp_path / "again", rows=TINY)
    a = json.loads((tiny / asof_data.MANIFEST).read_text())["tables"]
    b = json.loads((again / asof_data.MANIFEST).read_text())["tables"]
    assert a == b
    assert asof_data.verify(tiny) == []


def test_quote_timestamps_are_unique_within_the_day(tiny):
    quotes = pq.read_table(tiny / "quotes.parquet")
    ts = quotes.column("ts").cast("int64").to_pylist()
    assert len(set(ts)) == len(ts) == TINY["quotes"]
    assert min(ts) >= asof_data.START_US
    assert max(ts) < asof_data.START_US + asof_data.DAY_US
    # Shuffled within the chunk: not in time order.
    assert ts != sorted(ts)


def test_existing_dataset_is_never_overwritten(tiny):
    with pytest.raises(FileExistsError):
        asof_data.generate(1, tiny, rows=TINY)


def test_session_conf_enables_the_syntax(tiny):
    assert (tiny / "session.conf").read_text().strip() == "spark.sql.join.asofJoin.enabled=true"


def test_queries_load_per_engine():
    krishiv = {q.id: q.sql for q in load_queries("asof", "krishiv")}
    duck = {q.id: q.sql for q in load_queries("asof", "duckdb")}
    assert sorted(krishiv) == sorted(duck) == ["q01", "q02", "q03", "q04"]
    assert all("MATCH_CONDITION" in s for s in krishiv.values())
    assert not any("MATCH_CONDITION" in s for s in duck.values())


def test_duckdb_answers_match_a_brute_force_oracle(tiny):
    con = duckdb.connect()
    for t in ("trades", "quotes"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{tiny / (t + '.parquet')}')")
    trades = con.execute("SELECT ts, symbol, qty FROM trades").fetchall()
    quotes = con.execute("SELECT ts, symbol, bid_cents, ask_cents FROM quotes").fetchall()
    by_symbol: dict[str, list] = {}
    for q in quotes:
        by_symbol.setdefault(q[1], []).append(q)
    n = bid = qty = 0
    for ts, sym, q_qty in trades:
        cands = [q for q in by_symbol.get(sym, []) if q[0] <= ts]
        if cands:
            best = max(cands, key=lambda q: q[0])
            n += 1
            bid += best[2]
            qty += q_qty
    duck = {q.id: q.sql for q in load_queries("asof", "duckdb")}
    assert con.execute(duck["q01"]).fetchone() == (n, bid, qty)
    for name in ("q02", "q03", "q04"):
        row = con.execute(duck[name]).fetchone()
        assert row is not None, name
        assert row[0] > 0, name
