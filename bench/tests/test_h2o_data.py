"""db-benchmark data generator: row counts, cardinalities, NA shares, determinism, immutability, verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from krishiv_bench import h2o_data as hd

TINY_J1: dict[str, Any] = {
    "div_small": 1000,
    "div_medium": 100,
}  # 10 / 100 / 10_000 rows instead of 10 / 10_000 / 1e7 ...


def manifest(d: Path) -> dict:
    m = hd.read_manifest(d)
    assert m is not None
    return m


def q(path: Path, sql: str):
    """One row of ``sql`` where ``{f}`` is the parquet file."""
    return duckdb.connect().execute(sql.format(f=f"read_parquet('{path}')")).fetchone()


def test_sci_matches_the_upstream_pretty_sci():
    assert [hd.sci(n) for n in (10**7, 10**8, 10**9, 100, 10, 2)] == ["1e7", "1e8", "1e9", "1e2", "1e1", "2e0"]


def test_groupby_shape_cardinalities_and_ranges(tmp_path: Path):
    p = hd.G1Params(rows=20_000, k=10, chunk_rows=7_000)
    d = hd.generate("groupby", p, tmp_path)
    assert d.name == "G1_2e4_1e1_0_0"
    f = d / "x.parquet"
    t = pq.read_table(f)
    assert t.schema == hd.G1_SCHEMA
    assert t.num_rows == 20_000
    distinct = q(
        f, "SELECT count(DISTINCT id1), count(DISTINCT id2), count(DISTINCT id4), count(DISTINCT id5) FROM {f}"
    )
    assert distinct == (10, 10, 10, 10)
    big = q(f, "SELECT count(DISTINCT id3), count(DISTINCT id6) FROM {f}")  # 2000 groups, 20_000 draws each
    assert all(1900 < n <= 2000 for n in big)
    lo_hi = q(
        f,
        "SELECT min(id4), max(id4), max(id6), min(v1), max(v1), min(v2), max(v2), min(v3), max(v3), "
        "count(*) - count(id1) FROM {f}",
    )
    assert lo_hi[:2] == (1, 10)
    assert lo_hi[2] <= 2000
    assert lo_hi[3:7] == (1, 5, 1, 15)
    assert lo_hi[7] >= 0
    assert lo_hi[8] <= 100
    assert lo_hi[9] == 0  # no NAs asked for
    ids = q(f, "SELECT min(id1), max(id1), length(min(id3)), length(max(id3)) FROM {f}")
    assert ids == ("id001", "id010", 12, 12)  # id%03d and id%010d
    assert q(f, "SELECT count(*) FROM {f} WHERE v3 <> round(v3, 6)") == (0,)


def test_groupby_na_shares(tmp_path: Path):
    p = hd.G1Params(rows=10_000, k=10, na_pct=10, chunk_rows=4_000)
    f = hd.generate("groupby", p, tmp_path) / "x.parquet"
    # v1..v3: an exact share of every chunk (400 + 400 + 200 of 10_000)
    assert q(f, "SELECT count(*) - count(v1), count(*) - count(v2), count(*) - count(v3) FROM {f}") == (1000,) * 3
    # id columns: 10 % of the distinct values (1 of 10 for id1/id2/id4/id5, 100 of 1000 for id3/id6) are NA
    row = q(
        f,
        "SELECT count(DISTINCT id1), count(DISTINCT id3), count(DISTINCT id4), count(DISTINCT id6), "
        "count(*) - count(id1), count(*) - count(id3) FROM {f}",
    )
    assert row[:4] == (9, 900, 9, 900)
    assert 500 < row[4] < 1500
    assert 500 < row[5] < 1500


def test_groupby_is_deterministic_and_seed_sensitive(tmp_path: Path):
    p = hd.G1Params(rows=6_000, k=10, na_pct=5, chunk_rows=2_000)
    a = hd.generate("groupby", p, tmp_path / "a")
    b = hd.generate("groupby", p, tmp_path / "b")
    c = hd.generate("groupby", hd.G1Params(rows=6_000, k=10, na_pct=5, chunk_rows=2_000, seed=7), tmp_path / "c")
    ma, mb, mc = (manifest(x) for x in (a, b, c))
    assert ma["content_sha256"] == mb["content_sha256"]
    assert ma["tables"]["x"]["columns"] == mb["tables"]["x"]["columns"]
    assert ma["content_sha256"] != mc["content_sha256"]
    assert pq.read_table(a / "x.parquet").equals(pq.read_table(b / "x.parquet"))


def test_groupby_sorted_variant_is_the_same_rows_in_key_order(tmp_path: Path):
    kw: dict[str, Any] = {"rows": 6_000, "k": 10, "na_pct": 5, "chunk_rows": 2_000}
    u = hd.generate("groupby", hd.G1Params(**kw), tmp_path / "u") / "x.parquet"
    s_dir = hd.generate("groupby", hd.G1Params(**kw, sorted=True), tmp_path / "s")
    s = s_dir / "x.parquet"
    assert s_dir.name.endswith("_5_1")
    keys = "id1, id2, id3, id4, id5, id6"
    con = duckdb.connect()
    # same multiset of rows
    both = con.execute(
        f"SELECT count(*) FROM (SELECT * FROM read_parquet('{u}') EXCEPT ALL SELECT * FROM read_parquet('{s}'))"
    ).fetchone()
    assert both == (0,)
    rows = con.execute(f"SELECT {keys} FROM read_parquet('{s}')").fetchall()
    assert rows == sorted(rows, key=lambda r: tuple((x is not None, x) for x in r))  # NULLs first, then ascending
    assert manifest(s_dir)["content_sha256"] != manifest(tmp_path / "u" / "G1_6e3_1e1_5_0")["content_sha256"]
    # ties are broken by generation order, so the sorted file reproduces
    again = hd.generate("groupby", hd.G1Params(**kw, sorted=True), tmp_path / "s2")
    assert manifest(again)["content_sha256"] == manifest(s_dir)["content_sha256"]


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"rows": 0}, "rows must be >= 1"),
        ({"rows": 100, "k": 0}, "k must be in 1..999"),
        ({"rows": 100, "k": 1000}, "k must be in 1..999"),
        ({"rows": 101, "k": 10}, "multiple of k"),
        ({"rows": 100, "k": 10, "na_pct": 101}, "na_pct"),
        ({"rows": 100, "k": 10, "chunk_rows": 0}, "chunk_rows"),
    ],
)
def test_groupby_parameters_are_validated(tmp_path: Path, kwargs: dict, msg: str):
    with pytest.raises(ValueError, match=msg):
        hd.generate("groupby", hd.G1Params(**kwargs), tmp_path)
    assert not list(tmp_path.iterdir())


def test_task_and_params_must_agree(tmp_path: Path):
    with pytest.raises(ValueError, match="do not match task 'join'"):
        hd.generate("join", hd.G1Params(rows=100, k=10), tmp_path)


def test_join_tables_keys_and_match_share(tmp_path: Path):
    d = hd.generate("join", hd.J1Params(rows=10_000, chunk_rows=3_000, **TINY_J1), tmp_path)
    assert d.name == "J1_1e4_NA_0_0"
    n = {t: pq.read_table(d / f"{t}.parquet").num_rows for t in hd.J1_COLUMNS}
    assert n == {"x": 10_000, "small": 10, "medium": 100, "big": 10_000}
    for t, cols in hd.J1_COLUMNS.items():
        assert pq.read_table(d / f"{t}.parquet").schema.names == cols
    x, small, medium, big = (d / f"{t}.parquet" for t in ("x", "small", "medium", "big"))
    # every left key occurs, the right tables hold unique join keys
    assert q(x, "SELECT count(DISTINCT id1), count(DISTINCT id2), count(DISTINCT id3) FROM {f}") == (10, 100, 10_000)
    assert q(small, "SELECT count(DISTINCT id1) FROM {f}") == (10,)
    assert q(medium, "SELECT count(DISTINCT id2) FROM {f}") == (100,)
    assert q(big, "SELECT count(DISTINCT id3) FROM {f}") == (10_000,)
    # id4..id6 are the text forms of id1..id3
    assert q(x, "SELECT count(*) FROM {f} WHERE id4 <> 'id' || id1 OR id6 <> 'id' || id3") == (0,)
    # 90 % of the keys are common to both sides (upstream split_xlr), so exactly 90 % of the left rows match
    con = duckdb.connect()
    for rhs, key in ((small, "id1"), (medium, "id2"), (big, "id3")):
        got = con.execute(
            f"SELECT count(*) FROM read_parquet('{x}') x JOIN read_parquet('{rhs}') r ON x.{key} = r.{key}"
        ).fetchone()
        assert got is not None
        if rhs == big:
            assert got[0] == 9_000
        else:
            assert 0.5 * 10_000 < got[0] <= 10_000  # keys repeat unevenly, so only the key share is exact
    keys_in_both = con.execute(
        f"SELECT count(DISTINCT x.id1) FROM read_parquet('{x}') x JOIN read_parquet('{small}') s ON x.id1 = s.id1"
    ).fetchone()
    assert keys_in_both == (9,)


def test_join_na_only_in_the_left_table_and_deterministic(tmp_path: Path):
    p = hd.J1Params(rows=10_000, na_pct=10, chunk_rows=3_000, **TINY_J1)
    a = hd.generate("join", p, tmp_path / "a")
    b = hd.generate("join", p, tmp_path / "b")
    assert manifest(a)["content_sha256"] == manifest(b)["content_sha256"]
    x = a / "x.parquet"
    assert q(x, "SELECT count(DISTINCT id1), count(DISTINCT id2), count(DISTINCT id3) FROM {f}") == (9, 90, 9_000)
    assert q(x, "SELECT count(*) - count(v1) FROM {f}") == (1_000,)
    assert q(x, "SELECT count(*) FILTER (WHERE (id3 IS NULL) <> (id6 IS NULL)) FROM {f}") == (0,)  # id6 follows id3
    for t in ("small", "medium", "big"):
        nulls = q(
            a / f"{t}.parquet", "SELECT " + " + ".join(f"count(*) - count({c})" for c in hd.J1_COLUMNS[t]) + " FROM {f}"
        )
        assert nulls == (0,)


def test_join_sorted_variant(tmp_path: Path):
    d = hd.generate("join", hd.J1Params(rows=10_000, na_pct=10, sorted=True, chunk_rows=3_000, **TINY_J1), tmp_path)
    assert d.name == "J1_1e4_NA_10_1"
    con = duckdb.connect()
    for t, key in (("x", "id1, id2, id3"), ("small", "id1"), ("medium", "id2"), ("big", "id3")):
        rows = con.execute(f"SELECT {key} FROM read_parquet('{d / (t + '.parquet')}')").fetchall()
        assert rows == sorted(rows, key=lambda r: tuple((v is not None, v) for v in r)), t


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"rows": 10_000}, "multiple of div_small"),
        ({"rows": 10_005, **TINY_J1}, "multiple of div_small"),
        ({"rows": 10_000, "div_small": 1000, "div_medium": 3}, "multiple of div_medium"),
        ({"rows": 10_000, "na_pct": -1, **TINY_J1}, "na_pct"),
    ],
)
def test_join_parameters_are_validated(tmp_path: Path, kwargs: dict, msg: str):
    with pytest.raises(ValueError, match=msg):
        hd.generate("join", hd.J1Params(**kwargs), tmp_path)


def test_join_needs_ten_keys_per_table(tmp_path: Path):
    with pytest.raises(ValueError, match="multiple of 10"):
        hd.generate("join", hd.J1Params(rows=100, div_small=100, div_medium=10), tmp_path)  # 1 small key


def test_existing_dataset_is_never_overwritten(tmp_path: Path):
    p = hd.G1Params(rows=1_000, k=10)
    d = hd.generate("groupby", p, tmp_path)
    before = (d / "x.parquet").stat().st_mtime_ns
    with pytest.raises(FileExistsError, match="immutable"):
        hd.generate("groupby", p, tmp_path)
    assert (d / "x.parquet").stat().st_mtime_ns == before


def test_failed_generation_leaves_nothing_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(p):
        yield hd.g1_chunk(p, 0, 10)
        raise RuntimeError("disk full")

    monkeypatch.setattr(hd, "g1_chunks", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        hd.generate("groupby", hd.G1Params(rows=1_000, k=10), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_verify_accepts_the_generated_files_and_flags_changes(tmp_path: Path):
    d = hd.generate("groupby", hd.G1Params(rows=3_000, k=10, na_pct=5, chunk_rows=1_000), tmp_path)
    assert hd.verify(d) == []
    # digests do not depend on how the file is read back
    assert hd.digest_parquet(d / "x.parquet", batch_rows=7) == hd.digest_parquet(d / "x.parquet", batch_rows=100_000)
    t = pq.read_table(d / "x.parquet")
    v1 = t.column("v1").to_pylist()
    v1[0] = 6 if v1[0] != 6 else 7
    pq.write_table(t.set_column(t.schema.get_field_index("v1"), "v1", pa.array(v1, pa.int32())), d / "x.parquet")
    assert hd.verify(d) == ["x.v1: content digest differs from the manifest"]
    (d / "x.parquet").unlink()
    assert hd.verify(d) == [f"x: {d / 'x.parquet'} is missing"]
    assert hd.verify(tmp_path / "nope") == [f"{tmp_path / 'nope'}: no MANIFEST.json"]


def test_manifest_records_what_is_needed_to_reproduce(tmp_path: Path):
    d = hd.generate("groupby", hd.G1Params(rows=1_000, k=10), tmp_path)
    m = manifest(d)
    assert m["generator_version"] == hd.GENERATOR_VERSION
    assert m["task"] == "groupby"
    assert m["params"]["seed"] == hd.UPSTREAM_SEED
    assert m["params"]["rows"] == 1_000
    assert set(m["libs"]) == {"numpy", "pyarrow"}
    assert m["tables"]["x"]["rows"] == 1_000
    assert m["tables"]["x"]["file_sha256"] == hd.sha256_file(d / "x.parquet")
    assert set(m["tables"]["x"]["columns"]) == set(hd.G1_SCHEMA.names)
    assert m["content_sha256"] == hd.content_digest(m["tables"])


def test_digest_rejects_unsupported_types():
    with pytest.raises(TypeError, match="unsupported column type"):
        hd.ColumnDigest().update(pa.array([True]))
