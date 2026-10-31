"""Microbenchmark data: determinism, exact group and match structure, checksums, immutability, size-set validation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import duckdb
import pytest

from krishiv_bench import h2o_data
from krishiv_bench import micro_data as md

TINY = md.MicroSizes("tiny", 20_000, (10, 1_000), (100, 20_000), (100,))


def sql(path: Path, query: str):
    return duckdb.connect().execute(query.format(f=f"read_parquet('{path}')")).fetchall()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return md.generate(TINY, tmp_path_factory.mktemp("micro"))


def manifest(d: Path) -> dict:
    m = h2o_data.read_manifest(d)
    assert m is not None
    return m


def test_same_seed_same_checksum_and_other_seed_other(tiny: Path, tmp_path: Path):
    again = md.generate(TINY, tmp_path)
    assert manifest(again)["content_sha256"] == manifest(tiny)["content_sha256"]
    assert manifest(again)["tables"] == manifest(tiny)["tables"]  # also the Parquet file hashes (same libraries)
    other = md.generate(TINY, tmp_path / "other", seed=md.SEED + 1)
    assert manifest(other)["content_sha256"] != manifest(tiny)["content_sha256"]


def test_chunking_does_not_change_the_content(tiny: Path, tmp_path: Path, monkeypatch):
    # the digest is chunk independent but the random streams are per chunk: a different chunk size is a different
    # dataset, which is why CHUNK_ROWS is part of the generator definition and not a parameter.
    monkeypatch.setattr(md, "CHUNK_ROWS", 7_000)
    other = md.generate(TINY, tmp_path)
    assert manifest(other)["content_sha256"] != manifest(tiny)["content_sha256"]
    assert manifest(other)["tables"]["fact"]["rows"] == TINY.rows


def test_manifest_records_what_is_needed_to_reproduce(tiny: Path):
    m = manifest(tiny)
    assert m["generator_version"] == md.GENERATOR_VERSION
    assert m["params"]["seed"] == md.SEED
    assert md.sizes_from_params(m["params"]) == TINY
    assert set(m["tables"]) == {"fact", "build_1e1", "build_1e3"}
    assert m["tables"]["fact"]["rows"] == TINY.rows
    assert m["tables"]["fact"]["file_sha256"] == h2o_data.sha256_file(tiny / "fact.parquet")
    assert m["content_sha256"] == h2o_data.content_digest(m["tables"])
    assert h2o_data.verify(tiny) == []


def test_verify_notices_a_changed_file(tiny: Path, tmp_path: Path):
    copy = md.generate(TINY, tmp_path)
    assert h2o_data.verify(copy) == []
    duckdb.connect().execute(
        f"COPY (SELECT * REPLACE (v1 + 1 AS v1) FROM read_parquet('{copy / 'fact.parquet'}')) "
        f"TO '{tmp_path / 'edited.parquet'}' (FORMAT parquet)"
    )
    (tmp_path / "edited.parquet").replace(copy / "fact.parquet")
    assert any("fact.v1" in p for p in h2o_data.verify(copy))


def test_existing_dataset_is_never_overwritten(tiny: Path):
    with pytest.raises(FileExistsError, match="immutable"):
        md.generate(TINY, tiny.parent)
    assert h2o_data.verify(tiny) == []


def test_failed_generation_leaves_no_directory(tmp_path: Path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(md, "_write", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        md.generate(TINY, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unique_permutation_and_exact_group_counts(tiny: Path):
    f = tiny / "fact.parquet"
    assert sql(f, "SELECT COUNT(DISTINCT u), MIN(u), MAX(u), COUNT(*), COUNT(DISTINCT id) FROM {f}") == [
        (20_000, 0, 19_999, 20_000, 20_000)
    ]
    (row,) = sql(
        f, "SELECT COUNT(DISTINCT gi1e2), COUNT(DISTINCT gs1e2), COUNT(DISTINCT gj1e2), COUNT(DISTINCT gi2e4) FROM {f}"
    )
    assert row == (100, 100, 100, 20_000)
    # every group has the same size, and the pair of keys forms c * c groups
    assert sql(f, "SELECT MIN(n), MAX(n) FROM (SELECT COUNT(*) n FROM {f} GROUP BY gi1e2)") == [(200, 200)]
    assert sql(f, "SELECT COUNT(*), MIN(n), MAX(n) FROM (SELECT COUNT(*) n FROM {f} GROUP BY gi1e2, gj1e2)") == [
        (10_000, 2, 2)
    ]
    assert sql(f, "SELECT COUNT(*) FROM {f} WHERE gs1e2 <> 'id' || lpad(CAST(gi1e2 AS VARCHAR), 10, '0')") == [(0,)]


def test_selectivity_of_u_is_exact(tiny: Path):
    assert sql(tiny / "fact.parquet", "SELECT COUNT(*) FROM {f} WHERE u < 2000") == [(2_000,)]


def test_build_tables_hold_each_key_once(tiny: Path):
    for b in (10, 1_000):
        f = tiny / f"{md.build_table(b)}.parquet"
        assert sql(f, "SELECT COUNT(*), COUNT(DISTINCT k), MIN(k), MAX(k), COUNT(DISTINCT ks) FROM {f}") == [
            (b, b, 0, b - 1, b)
        ]
        assert sql(f, "SELECT COUNT(*) FROM {f} WHERE ks <> 'id' || lpad(CAST(k AS VARCHAR), 10, '0')") == [(0,)]
        assert sql(f, "SELECT COUNT(*) FROM {f} WHERE bv <> k * 7919 % 100003") == [(0,)]


@pytest.mark.parametrize("pct", md.MATCH_PCTS_INT)
def test_match_rate_follows_the_requested_share(tiny: Path, pct: int):
    q = (
        f"SELECT COUNT(*) FROM read_parquet('{tiny / 'fact.parquet'}') f "
        f"JOIN read_parquet('{tiny / 'build_1e3.parquet'}') b ON f.{md.col_ji(1_000, pct)} = b.k"
    )
    (n,) = duckdb.connect().execute(q).fetchone()  # ty: ignore[not-iterable]
    # each probe row matches with probability pct/100 (binomial, sd 0.4% of the rows at worst)
    assert abs(n - TINY.rows * pct / 100) < 0.02 * TINY.rows


def test_text_keys_hit_exactly_where_the_int_keys_do(tiny: Path):
    for pct in md.MATCH_PCTS_STR:
        assert sql(
            tiny / "fact.parquet",
            f"SELECT COUNT(*) FROM {{f}} WHERE {md.col_js(1_000, pct)} <> "
            f"'id' || lpad(CAST({md.col_ji(1_000, pct)} AS VARCHAR), 10, '0')",
        ) == [(0,)]


def test_zipf_keys_stay_in_the_build_side_and_are_skewed(tiny: Path):
    f = tiny / "fact.parquet"
    for alpha in md.ZIPF_ALPHAS:
        c = md.col_jz(1_000, alpha)
        assert sql(f, f"SELECT MIN({c}) >= 0, MAX({c}) < 1000 FROM {{f}}") == [(True, True)]
    hot08, hot12 = (sql(f, f"SELECT COUNT(*) FROM {{f}} WHERE {md.col_jz(1_000, a)} = 0")[0][0] for a in md.ZIPF_ALPHAS)
    assert hot12 > hot08 > TINY.rows / 1_000 * 5  # the hottest key is far above a uniform share, more so at alpha 1.2


def test_needle_rows_and_value_ranges(tiny: Path):
    f = tiny / "fact.parquet"
    assert sql(f, f"SELECT COUNT(*) FROM {{f}} WHERE s = '{md.NEEDLE}'") == [(TINY.rows // md.NEEDLE_EVERY,)]
    assert sql(f, "SELECT COUNT(*) FROM {f} WHERE length(s) <> 16 OR s NOT SIMILAR TO '[a-z0-9]+'") == [(0,)]
    assert sql(f, "SELECT MIN(v1), MAX(v1), MIN(v3) >= 0, MAX(v3) < 100 FROM {f}") == [(1, 100, True, True)]
    ((typ, lo, hi),) = sql(f, "SELECT typeof(MIN(dec)), MIN(dec), MAX(dec) FROM {f}")
    assert typ == "DECIMAL(15,2)"
    assert lo < 0 < hi  # both signs reach the 128-bit words


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rows": 999}, "rows must be a multiple of 1000"),
        ({"rows": 10_500}, "rows must be a multiple of 1000"),
        ({"builds": ()}, "builds must be distinct"),
        ({"builds": (5,)}, "builds must be distinct"),
        ({"builds": (100, 100)}, "builds must be distinct"),
        ({"groups": ()}, "groups must be distinct"),
        ({"groups": (10_001,)}, "groups must be distinct"),
        ({"pairs": (50,)}, "pairs must be group counts"),
        ({"pairs": (10_000,), "groups": (10_000,)}, "pairs must be group counts"),
    ],
)
def test_size_set_validation(kwargs: dict, message: str):
    base = md.MicroSizes("x", 10_000, (10,), (100,))
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **kwargs).validate()


def test_named_size_sets_are_valid_and_ordered():
    for p in md.SIZES.values():
        p.validate()
    assert list(md.SIZES) == ["quick", "default", "full"]
    assert md.SIZES["quick"].rows < md.SIZES["default"].rows < md.SIZES["full"].rows
    assert max(md.SIZES["full"].builds) == 10**8
    assert max(md.SIZES["full"].groups) == 10**8


def test_column_names_are_stable():
    assert md.col_gi(1_000_000) == "gi1e6"
    assert md.col_ji(100_000, 50) == "ji_b1e5_m50"
    assert md.col_jz(1_000, 1.2) == "jz_b1e3_a12"
    assert md.col_jz(1_000, 0.8) == "jz_b1e3_a08"
    assert md.build_table(10**8) == "build_1e8"


def test_cli_generates_verifies_and_refuses_to_overwrite(tmp_path: Path, monkeypatch, capsys):
    from krishiv_bench import datasets

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    monkeypatch.setitem(md.SIZES, "quick", TINY)
    datasets.main(["micro", "--size", "quick"])
    d = tmp_path / "micro" / md.GENERATOR_VERSION / "tiny"
    assert (d / "fact.parquet").is_file()
    assert f"content sha256 {manifest(d)['content_sha256']}" in capsys.readouterr().out
    datasets.main(["micro-verify", str(d)])
    assert "verified" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="immutable"):
        datasets.main(["micro", "--size", "quick"])
    datasets.main(["micro", "--size", "quick", "--micro-version", "v2"])  # a new version is a new directory
    assert (tmp_path / "micro" / "v2" / "tiny" / "fact.parquet").is_file()
