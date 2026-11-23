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
    trades = con.execute("SELECT ts, symbol, qty, price_cents FROM trades").fetchall()
    quotes = con.execute("SELECT ts, symbol, bid_cents, ask_cents FROM quotes").fetchall()
    by_symbol: dict[str, list] = {}
    for q in quotes:
        by_symbol.setdefault(q[1], []).append(q)
    # Each query's exact answer, by brute force: the nearest quote per trade in the query's direction.
    q01 = [0, 0, 0]  # backward, same symbol: n, sum(bid), sum(qty)
    q02 = [0, 0, 0]  # strictly forward, same symbol: n, sum(ask), sum(price)
    q03 = [0, 0, 0]  # backward LEFT over symbols < S0900: n (every trade), matched, sum(bid)
    small = [q for q in quotes if q[1] < "S0100"]
    q04 = [0, 0]  # backward with no key, both sides symbols < S0100: n, sum(bid)
    for ts, sym, qty, price in trades:
        same = by_symbol.get(sym, [])
        back = [q for q in same if q[0] <= ts]
        if back:
            best = max(back, key=lambda q: q[0])
            q01[0] += 1
            q01[1] += best[2]
            q01[2] += qty
        fwd = [q for q in same if q[0] > ts]
        if fwd:
            best = min(fwd, key=lambda q: q[0])
            q02[0] += 1
            q02[1] += best[3]
            q02[2] += price
        q03[0] += 1
        if back and sym < "S0900":
            q03[1] += 1
            q03[2] += max(back, key=lambda q: q[0])[2]
        if sym < "S0100":
            cands = [q for q in small if q[0] <= ts]
            if cands:
                q04[0] += 1
                q04[1] += max(cands, key=lambda q: q[0])[2]  # quote timestamps are unique: no tie
    duck = {q.id: q.sql for q in load_queries("asof", "duckdb")}
    # An empty match sums to NULL; the tiny dataset matches every variant, so no NULL is expected.
    assert all(v > 0 for v in (q01[0], q02[0], q03[1], q04[0]))
    assert con.execute(duck["q01"]).fetchone() == tuple(q01)
    assert con.execute(duck["q02"]).fetchone() == tuple(q02)
    assert con.execute(duck["q03"]).fetchone() == tuple(q03)
    assert con.execute(duck["q04"]).fetchone() == tuple(q04)


def test_a_failed_generation_leaves_nothing_behind(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(asof_data, "_write", boom)
    with pytest.raises(RuntimeError):
        asof_data.generate(1, tmp_path / "ds", rows=TINY)
    assert list(tmp_path.iterdir()) == []


def test_a_dataset_published_meanwhile_is_not_replaced(tmp_path, monkeypatch):
    final = tmp_path / "ds"
    real = asof_data._write

    def racing(sf, rows, tmp):
        real(sf, rows, tmp)
        final.mkdir()
        (final / "other").write_text("x")  # another generator finished first

    monkeypatch.setattr(asof_data, "_write", racing)
    with pytest.raises(FileExistsError):
        asof_data.generate(1, final, rows=TINY)
    assert [p.name for p in tmp_path.iterdir()] == ["ds"]
    assert [p.name for p in final.iterdir()] == ["other"]


def test_run_suite_has_a_default_dataset():
    from krishiv_bench.cli import ABBA_DATASET
    from krishiv_bench.runner import SUITE_DATASET

    assert SUITE_DATASET["asof"] == ABBA_DATASET["asof"] == "asof"


def test_rows_of_refuses_a_non_positive_scale():
    assert asof_data.rows_of(2) == {"trades": 2 * asof_data.TRADES_PER_SF, "quotes": 2 * asof_data.QUOTES_PER_SF}
    with pytest.raises(ValueError, match="positive"):
        asof_data.rows_of(0)


def test_verify_names_a_row_count_or_digest_the_manifest_disagrees_with(tiny, tmp_path):
    import shutil

    d = tmp_path / "copy"
    shutil.copytree(tiny, d)
    path = d / asof_data.MANIFEST
    man = json.loads(path.read_text())
    man["tables"]["trades"]["rows"] += 1
    man["tables"]["quotes"]["content_sha256"] = "0" * 64
    path.write_text(json.dumps(man))
    problems = asof_data.verify(d)
    assert any(p.startswith("trades:") and "rows" in p for p in problems), problems
    assert "quotes: content digest differs from the manifest" in problems


def test_cli_generates_verifies_and_refuses(tmp_path, monkeypatch, capsys):
    from krishiv_bench import datasets

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    real = asof_data.rows_of
    monkeypatch.setattr(asof_data, "rows_of", lambda sf: TINY if sf > 0 else real(sf))
    datasets.main(["asof", "--sf", "1"])
    d = asof_data.asof_dir(1)
    assert d == tmp_path / "asof" / "sf1"
    assert "manifest:" in capsys.readouterr().out
    datasets.main(["asof-verify", str(d)])
    assert "verified" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        datasets.main(["asof", "--sf", "1"])  # an existing dataset is never overwritten
    with pytest.raises(SystemExit):
        datasets.main(["asof", "--sf", "0"])
    path = d / asof_data.MANIFEST
    man = json.loads(path.read_text())
    man["tables"]["trades"]["content_sha256"] = "0" * 64
    path.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="trades: content digest"):
        datasets.main(["asof-verify", str(d)])
