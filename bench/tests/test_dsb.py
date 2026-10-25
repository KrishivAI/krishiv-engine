"""DSB dataset: schema parsing, query ids, relation counts, manifest verification, harness hooks (no generator)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from krishiv_bench import compare, dsb, runner

DDL = """
-- comment
create table dbgen_version
(
    dv_version                varchar(16)                   ,
    dv_create_date            date
);

create table warehouse
(
    w_warehouse_sk            integer               not null,
    w_warehouse_name          varchar(20)                   ,
    w_state                   char(2)                       ,
    w_gmt_offset              decimal(5,2)                  ,
    w_opened                  date                          ,
    w_at                      time                          ,
    primary key (w_warehouse_sk)
);
"""


def test_parse_schema_maps_types_and_skips_dbgen_version():
    schema = dsb.parse_schema(DDL)
    assert list(schema) == ["warehouse"]
    assert schema["warehouse"] == [
        ("w_warehouse_sk", "INTEGER"),
        ("w_warehouse_name", "VARCHAR"),
        ("w_state", "VARCHAR"),
        ("w_gmt_offset", "DECIMAL(5,2)"),
        ("w_opened", "DATE"),
        ("w_at", "TIME"),
    ]


def test_parse_schema_rejects_unknown_types():
    with pytest.raises(ValueError, match="unsupported type"):
        dsb.parse_schema("create table t\n(\n    a   geometry\n);")


def test_template_files_ids(tmp_path: Path):
    for d in dsb.TEMPLATE_DIRS.values():
        (tmp_path / "query_templates_pg" / d).mkdir(parents=True)
        (tmp_path / "query_templates_pg" / d / "postgres.tpl").write_text("")
    (tmp_path / "query_templates_pg" / "agg_queries" / "query013.tpl").write_text("")
    (tmp_path / "query_templates_pg" / "spj_queries" / "query013_spj.tpl").write_text("")
    (tmp_path / "query_templates_pg" / "multi_block_queries" / "query001.tpl").write_text("")
    assert list(dsb.template_files(tmp_path)) == ["agg_q013", "mb_q001", "spj_q013"]


def test_relation_count_counts_every_base_table_reference():
    sql = "select 1 from a join b on a.x = b.x join a a2 on a2.x = b.x where a.y in (select y from c)"
    assert dsb.relation_count(sql) == 4
    assert dsb.relation_count("select 1") == 0
    assert dsb.relation_count("this is not sql") is None


def test_count_distribution_ignores_unparsed():
    assert dsb.count_distribution({"a": 3, "b": 3, "c": 10, "d": None}) == {3: 2, 10: 1}


def make_dataset(root: Path) -> dict:
    (root / "queries").mkdir(parents=True)
    (root / "queries" / "agg_q013.sql").write_text("select 1;\n")
    (root / "t.parquet").write_bytes(b"parquet")
    dsb.save_oracle(root / dsb.ORACLE, {"agg_q013": [(1,)]})
    man = {
        "tables": {"t": {"file": "t.parquet", "file_sha256": hashlib.sha256(b"parquet").hexdigest()}},
        "queries": {"agg_q013": {"sha256": hashlib.sha256(b"select 1;\n").hexdigest()}},
        "oracle_file": dsb.ORACLE,
        "oracle_file_sha256": dsb.sha256_file(root / dsb.ORACLE),
    }
    (root / dsb.MANIFEST).write_text(json.dumps(man))
    return man


def test_verify_accepts_a_consistent_dataset_and_reports_each_corruption(tmp_path: Path):
    make_dataset(tmp_path)
    assert dsb.verify(tmp_path) == []
    (tmp_path / "t.parquet").write_bytes(b"other")
    (tmp_path / "queries" / "agg_q013.sql").write_text("select 2;\n")
    (tmp_path / dsb.ORACLE).unlink()
    problems = dsb.verify(tmp_path)
    assert len(problems) == 3
    assert any("Parquet sha256" in p for p in problems)
    assert any("query text" in p for p in problems)
    assert any("missing" in p for p in problems)


def test_verify_without_manifest(tmp_path: Path):
    assert "no MANIFEST.json" in dsb.verify(tmp_path)[0]


def test_load_queries_and_oracle(tmp_path: Path):
    make_dataset(tmp_path)
    (tmp_path / "queries" / "mb_q001.sql").write_text("select 2;\n")
    assert [q.id for q in dsb.load_queries(tmp_path)] == ["agg_q013", "mb_q001"]
    assert [q.id for q in dsb.load_queries(tmp_path, ["mb_q001"])] == ["mb_q001"]
    assert dsb.load_oracle(tmp_path) == {"agg_q013": [(1,)]}
    assert dsb.load_oracle(tmp_path / "nowhere") == {}
    with pytest.raises(SystemExit, match="no DSB queries"):
        dsb.load_queries(tmp_path / "nowhere")


def test_runner_loads_dsb_queries_from_the_dataset_only(tmp_path: Path):
    make_dataset(tmp_path)
    assert [q.suite for q in runner.load_queries("dsb", "duckdb", None, tmp_path)] == ["dsb"]
    with pytest.raises(SystemExit, match="dataset directory"):
        runner.load_queries("dsb", "duckdb")


def test_manifest_licence_names_the_tpc_terms():
    assert "TPC End User License Agreement" in dsb.LICENCE["generator_and_templates"]
    assert "not comparable" in dsb.LICENCE["benchmark"]


def test_convert_table_roundtrip(tmp_path: Path):
    import datetime
    from decimal import Decimal

    import duckdb
    import pyarrow.parquet as pq

    dat = tmp_path / "w.dat"
    dat.write_bytes("1|th\xe9|2.50||\n2||0.00|2001-02-03|04:05:06\n".encode("latin-1"))
    cols = [
        ("a", "INTEGER"),
        ("b", "VARCHAR"),
        ("c", "DECIMAL(5,2)"),
        ("d", "DATE"),
        ("e", "TIME"),
    ]
    out = tmp_path / "w.parquet"
    rec = dsb.convert_table(dat, cols, out, threads=1)
    assert rec["rows"] == 2
    assert rec["file"] == "w.parquet"
    t = sorted(pq.read_table(out).to_pylist(), key=lambda r: r["a"])
    assert t == [
        {"a": 1, "b": "thé", "c": Decimal("2.50"), "d": None, "e": None},
        {"a": 2, "b": None, "c": Decimal("0.00"), "d": datetime.date(2001, 2, 3), "e": datetime.time(4, 5, 6)},
    ]
    assert rec["file_sha256"] == dsb.sha256_file(out)
    expected = (
        duckdb.connect()
        .execute(
            f"SELECT coalesce(sum(h::HUGEINT), 0)::VARCHAR FROM (SELECT hash(t) AS h FROM read_parquet('{out}') t)"
        )
        .fetchone()[0]
    )
    assert rec["row_hash_sum"] == expected


def test_oracle_file_is_reproducible_and_loadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows = {"q": [(1, "a", None)]}
    dsb.save_oracle(tmp_path / "a.gz", rows)
    monkeypatch.setattr("time.time", lambda: 4e9)
    dsb.save_oracle(tmp_path / "b.gz", rows)
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()
    assert compare.load_rows(tmp_path / "a.gz") == rows
