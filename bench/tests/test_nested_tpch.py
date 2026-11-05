# SPDX-License-Identifier: Apache-2.0
"""Nested TPC-H (`krishiv_bench.nested_tpch`): lossless conversion, a verifier that catches damage, 22 queries."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pytest

from krishiv_bench import nested_tpch as n
from krishiv_bench.paths import queries_dir
from krishiv_bench.runner import load_queries

ROWS = 7


def _value(ty: str, col: str, i: int) -> str:
    if ty == "VARCHAR":
        return f"'{col}-{i}'"
    if ty == "DATE":
        return f"DATE '1995-01-{1 + i:02d}'"
    if ty.startswith("DECIMAL"):
        return f"CAST({i * 1000 + 7} AS DECIMAL(15,2)) / 100"
    return str(i * 3 + 1)


@pytest.fixture
def flat(tmp_path: Path) -> Path:
    """A tiny flat dataset in the shapes a generator writes (wider decimals and ints than the nested casts)."""
    root = tmp_path / "flat"
    root.mkdir()
    con = duckdb.connect()
    for t in n.TABLES:
        rows = " UNION ALL ".join(
            "SELECT " + ", ".join(f"{_value(ty, c, i)} AS {c}" for c, ty in (*t.top, *t.nested)) for i in range(ROWS)
        )
        con.execute(f"COPY ({rows}) TO '{root / (t.name + '.parquet')}' (FORMAT parquet)")
    return root


def test_the_tables_match_the_tpch_schema_names():
    names = {c for t in n.TABLES for c, _ in (*t.top, *t.nested)}
    assert len(names) == sum(len(t.top) + len(t.nested) for t in n.TABLES)  # no column is listed twice
    assert [t.name for t in n.TABLES] == [
        "region",
        "nation",
        "supplier",
        "part",
        "partsupp",
        "customer",
        "orders",
        "lineitem",
    ]
    # Keys and the partition columns stay on top, as Comet's conversion has them.
    top = {c for t in n.TABLES for c, _ in t.top}
    assert {"p_brand", "c_mktsegment", "o_orderdate", "l_shipdate"} <= top
    assert all(c.endswith("key") or c in {"p_brand", "c_mktsegment", "o_orderdate", "l_shipdate"} for c in top)


def test_nesting_keeps_every_row_and_wraps_the_columns_in_one_element(flat: Path, tmp_path: Path):
    out = tmp_path / "nested"
    rows = n.nest(flat, out)
    assert rows == {t.name: ROWS for t in n.TABLES}
    con = duckdb.connect()
    desc = dict(
        (r[0], r[1])
        for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out / 'lineitem.parquet'}')").fetchall()
    )
    assert desc["l_orderkey"] == "BIGINT"
    assert desc["l_shipdate"] == "DATE"
    assert desc["lineitem_data"].startswith("STRUCT(")
    assert desc["lineitem_data"].endswith(")[]")
    assert "l_quantity DECIMAL(12,2)" in desc["lineitem_data"]
    assert n.verify(flat, out) == []


def test_verify_catches_a_changed_value_a_lost_row_and_a_longer_array(flat: Path, tmp_path: Path):
    out = tmp_path / "nested"
    n.nest(flat, out)
    con = duckdb.connect()
    lineitem = out / "lineitem.parquet"
    assert n.scalar(con, f"SELECT count(*) FROM read_parquet('{lineitem}')") == ROWS

    def damaged(sql: str) -> list[str]:
        d = tmp_path / "damaged"
        d.mkdir(exist_ok=True)
        for t in n.TABLES:
            (d / f"{t.name}.parquet").write_bytes((out / f"{t.name}.parquet").read_bytes())
        con.execute(f"COPY ({sql.format(src=lineitem)}) TO '{d / 'lineitem.parquet'}' (FORMAT parquet)")
        return n.verify(flat, d)

    fields = ", ".join(
        f"{c} := " + ("'tampered'" if c == "l_comment" else f"lineitem_data[1].{c}")
        for c, _ in n.BY_NAME["lineitem"].nested
    )
    changed = damaged(
        "SELECT l_orderkey, l_partkey, l_suppkey, l_shipdate, "
        f"[struct_pack({fields})] AS lineitem_data FROM read_parquet('{{src}}')"
    )
    assert any("lineitem" in p and "rows" in p for p in changed), changed
    lost = damaged("SELECT * FROM read_parquet('{src}') LIMIT 6")
    assert any("6 nested rows, 7 flat rows" in p for p in lost), lost
    longer = damaged(
        "SELECT l_orderkey, l_partkey, l_suppkey, l_shipdate, "
        "list_concat(lineitem_data, lineitem_data) AS lineitem_data "
        "FROM read_parquet('{src}')"
    )
    assert any("does not hold exactly one element" in p for p in longer), longer


def test_the_digest_does_not_depend_on_the_parquet_encoding(flat: Path, tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    n.nest(flat, a)
    n.nest(flat, b)
    assert n.digest(a) == n.digest(b)
    con = duckdb.connect()
    src = b / "region.parquet"
    other = tmp_path / "region.parquet"
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{src}')) TO '{other}' "
        "(FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE 2)"
    )
    other.replace(src)
    assert n.digest(a) == n.digest(b)


def test_the_flat_source_may_be_a_directory_of_part_files(flat: Path, tmp_path: Path):
    parts = tmp_path / "parts"
    parts.mkdir()
    con = duckdb.connect()
    for t in n.TABLES:
        d = parts / t.name
        d.mkdir()
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{flat / (t.name + '.parquet')}')) "
            f"TO '{d / 'part-0.parquet'}' (FORMAT parquet)"
        )
    out = tmp_path / "nested"
    assert n.nest(parts, out) == {t.name: ROWS for t in n.TABLES}
    assert n.verify(parts, out) == []
    with pytest.raises(FileNotFoundError):
        n.source_path(tmp_path / "nowhere", "lineitem")


def test_the_22_queries_load_and_explode_only_nested_columns():
    qs = load_queries("tpch-nested", "krishiv")
    assert [q.id for q in qs] == [f"q{i:02d}" for i in range(1, 23)]
    for q in qs:
        assert "-- CometBench-H nested query" in q.sql, q.id
        assert re.search(r"\bexplode\(\w+_data\)", q.sql), f"{q.id} explodes nothing"
        used = set(re.findall(r"explode\((\w+_data)\)", q.sql))
        assert used <= {t.struct for t in n.TABLES}, (q.id, used)
    assert (queries_dir("tpch-nested") / "q01.sql").read_text().startswith("-- CometBench-H nested query 1")


class _Fake:
    """An engine stand-in: answers a query id with the given rows (the DuckDB side answers from a table)."""

    def __init__(self, answers):
        self.answers = answers

    def setup(self, ds):
        self.ds = ds

    def run(self, q):
        return self.answers[q.id]

    def teardown(self):
        pass


def _check(monkeypatch, tmp_path, nested_rows):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    flat_answers = {f"q{i:02d}": [(i, "a", 1.5)] for i in range(1, 23)}
    engines = {"duckdb": _Fake(flat_answers), "krishiv": _Fake({**flat_answers, **nested_rows})}
    return n.check("krishiv", 1, ["q01", "q02"], make=lambda name, **kw: engines[name])


def test_check_passes_when_the_nested_answer_equals_the_flat_one(monkeypatch, tmp_path):
    assert _check(monkeypatch, tmp_path, {}) == []


def test_check_reports_a_wrong_answer_and_an_engine_error(monkeypatch, tmp_path):
    bad = _check(monkeypatch, tmp_path, {"q01": [(1, "a", 2.5)]})
    assert len(bad) == 1
    assert bad[0].startswith("q01:")

    class Boom(_Fake):
        def run(self, q):
            raise RuntimeError("explode is not supported")

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    flat = {f"q{i:02d}": [(i, "a", 1.5)] for i in range(1, 23)}
    engines = {"duckdb": _Fake(flat), "spark-connect": Boom({})}
    bad = n.check("spark-connect", 1, ["q03"], make=lambda name, **kw: engines[name])
    assert bad == ["q03: spark-connect failed: explode is not supported"]


def test_engines_that_cannot_run_explode_are_refused_for_the_suite():
    for engine in ("duckdb", "polars", "datafusion", "clickhouse"):
        with pytest.raises(SystemExit, match="Spark SQL"):
            load_queries("tpch-nested", engine)
