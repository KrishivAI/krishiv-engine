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
    row = (
        duckdb.connect()
        .execute(
            f"SELECT coalesce(sum(h::HUGEINT), 0)::VARCHAR FROM (SELECT hash(t) AS h FROM read_parquet('{out}') t)"
        )
        .fetchone()
    )
    assert row is not None
    assert rec["row_hash_sum"] == row[0]


def test_oracle_file_is_reproducible_and_loadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows = {"q": [(1, "a", None)]}
    dsb.save_oracle(tmp_path / "a.gz", rows)
    monkeypatch.setattr("time.time", lambda: 4e9)
    dsb.save_oracle(tmp_path / "b.gz", rows)
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()
    assert compare.load_rows(tmp_path / "a.gz") == rows


def test_cache_dir_is_pinned_under_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    assert dsb.cache_dir() == tmp_path / "cache" / "dsb" / dsb.COMMIT


def test_run_invokes_subprocess_with_check(monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    seen: dict = {}

    def _fake(cmd, *args, **kwargs):
        seen["cmd"] = cmd
        seen["check"] = kwargs.get("check")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake)
    dsb._run(["echo", "hi"])
    assert seen == {"cmd": ["echo", "hi"], "check": True}


def test_fetch_source_returns_existing_checkout_at_pinned_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path / "cache")
    (tmp_path / "cache" / "src" / ".git").mkdir(parents=True)
    seen: list = []

    def _fake(cmd, *args, **kwargs):
        seen.append(cmd)
        return SimpleNamespace(stdout=dsb.COMMIT + "\n")

    monkeypatch.setattr(subprocess, "run", _fake)
    assert dsb.fetch_source() == tmp_path / "cache" / "src"
    assert all("init" not in cmd for cmd in seen)


def test_fetch_source_reclones_a_stale_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path / "cache")
    (tmp_path / "cache" / "src" / ".git").mkdir(parents=True)
    calls = {"rev_parse": 0}

    def _fake(cmd, *args, **kwargs):
        if "rev-parse" in cmd:
            calls["rev_parse"] += 1
            out = "deadbeef\n" if calls["rev_parse"] == 1 else dsb.COMMIT + "\n"
            return SimpleNamespace(stdout=out)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", _fake)
    assert dsb.fetch_source() == tmp_path / "cache" / "src"
    assert calls["rev_parse"] == 2


def test_fetch_source_clones_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path / "cache")
    seen: list = []

    def _fake(cmd, *args, **kwargs):
        seen.append(cmd)
        if "rev-parse" in cmd:
            return SimpleNamespace(stdout=dsb.COMMIT + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(subprocess, "run", _fake)
    assert dsb.fetch_source() == tmp_path / "cache" / "src"
    kinds = [cmd[1] for cmd in seen if cmd[0] == "git"]
    assert "init" in kinds
    assert "fetch" in kinds
    assert "checkout" in kinds


def test_fetch_source_rejects_an_unexpected_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="deadbeef\n"))
    with pytest.raises(RuntimeError, match="expected"):
        dsb.fetch_source()


def test_build_stamp_records_gcc_and_flags(monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="gcc (GCC) 15.2.1 20251105\n"))
    assert dsb._build_stamp() == {"gcc": "gcc (GCC) 15.2.1 20251105", "tool_cflags": dsb.TOOL_CFLAGS}


def test_build_stamp_without_gcc_output(monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    assert dsb._build_stamp()["gcc"] == "unknown"


def _write_tool_sources(src: Path) -> None:
    tools = src / "code" / "tools"
    tools.mkdir(parents=True)
    (tools / "Makefile.suite").write_text("all:\n")
    for name in ("qgen.y", "tokenizer.l", "y.tab.c", "y.tab.h", "tokenizer.c", "lex.yy.c"):
        (tools / name).write_text("/* fake */\n")


def test_build_tools_reuses_a_matching_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil
    import subprocess

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(dsb, "_build_stamp", lambda: {"gcc": "gcc-test", "tool_cflags": "flags"})
    monkeypatch.setattr(shutil, "which", lambda *a, **k: "/usr/bin/tool")
    build = tmp_path / "build"
    build.mkdir(parents=True)
    (build / "dsdgen").write_bytes(b"x")
    (build / "dsqgen").write_bytes(b"x")
    (build / ".krishiv-build-stamp.json").write_text(json.dumps(dsb._build_stamp(), sort_keys=True) + "\n")

    def _boom(*args, **kwargs):
        raise AssertionError("a matching stamp must not rebuild")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert dsb.build_tools(tmp_path / "src") == build


def test_build_tools_requires_gcc_and_make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="gcc"):
        dsb.build_tools(tmp_path / "src")


def test_build_tools_rebuilds_on_a_stale_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(dsb, "_build_stamp", lambda: {"gcc": "gcc-test", "tool_cflags": "flags"})
    monkeypatch.setattr(shutil, "which", lambda *a, **k: "/usr/bin/tool")
    src = tmp_path / "src"
    _write_tool_sources(src)
    build = tmp_path / "build"
    build.mkdir(parents=True)
    (build / "dsdgen").write_bytes(b"stale")
    (build / "dsqgen").write_bytes(b"stale")
    (build / ".krishiv-build-stamp.json").write_text(json.dumps({"gcc": "old", "tool_cflags": "old"}) + "\n")

    def _fake_make(cmd, *args, **kwargs):
        assert cmd[0] == "make"
        out = Path(kwargs["cwd"])
        (out / "dsdgen").write_bytes(b"new")
        (out / "dsqgen").write_bytes(b"new")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_make)
    assert dsb.build_tools(src) == build
    assert (build / "dsdgen").read_bytes() == b"new"
    assert json.loads((build / ".krishiv-build-stamp.json").read_text()) == dsb._build_stamp()


def test_build_tools_rebuilds_on_a_corrupt_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(dsb, "_build_stamp", lambda: {"gcc": "gcc-test", "tool_cflags": "flags"})
    monkeypatch.setattr(shutil, "which", lambda *a, **k: "/usr/bin/tool")
    src = tmp_path / "src"
    _write_tool_sources(src)
    build = tmp_path / "build"
    build.mkdir(parents=True)
    (build / "dsdgen").write_bytes(b"x")
    (build / "dsqgen").write_bytes(b"x")
    (build / ".krishiv-build-stamp.json").write_text("not json")

    def _fake_make(cmd, *args, **kwargs):
        out = Path(kwargs["cwd"])
        (out / "dsdgen").write_bytes(b"x")
        (out / "dsqgen").write_bytes(b"x")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_make)
    assert dsb.build_tools(src) == build


def test_build_tools_reports_when_make_produces_no_binaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(dsb, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(dsb, "_build_stamp", lambda: {"gcc": "gcc-test", "tool_cflags": "flags"})
    monkeypatch.setattr(shutil, "which", lambda *a, **k: "/usr/bin/tool")
    src = tmp_path / "src"
    _write_tool_sources(src)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="produced no"):
        dsb.build_tools(src)


def test_parse_schema_rejects_an_unparseable_column():
    with pytest.raises(ValueError, match="cannot parse column"):
        dsb.parse_schema("create table t\n(\n    justaname\n);")


def test_generate_queries_instantiates_every_template_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    build = tmp_path / "build"
    build.mkdir()
    src = tmp_path / "src"
    agg = src / "query_templates_pg" / "agg_queries"
    agg.mkdir(parents=True)
    (agg / "query001.tpl").write_text("SELECT 1")

    def _fake_dsqgen(cmd, *args, **kwargs):
        assert cmd[0].endswith("dsqgen")
        work = Path(cmd[cmd.index("-OUTPUT_DIR") + 1])
        tpl = cmd[cmd.index("-TEMPLATE") + 1]
        (work / "query_0.sql").write_text(f"-- {tpl}\nSELECT 1;\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_dsqgen)
    assert dsb.generate_queries(build, src, 1, 7, tmp_path / "queries") == {"agg_q001": "-- query001.tpl\nSELECT 1;\n"}


def test_generate_queries_rejects_empty_dsqgen_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    build = tmp_path / "build"
    build.mkdir()
    src = tmp_path / "src"
    agg = src / "query_templates_pg" / "agg_queries"
    agg.mkdir(parents=True)
    (agg / "query001.tpl").write_text("SELECT 1")

    def _fake_empty(cmd, *args, **kwargs):
        work = Path(cmd[cmd.index("-OUTPUT_DIR") + 1])
        (work / "query_0.sql").write_text("   \n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_empty)
    with pytest.raises(RuntimeError, match="no text"):
        dsb.generate_queries(build, src, 1, 7, tmp_path / "queries")


def test_relation_count_returns_none_when_serialization_raises(monkeypatch: pytest.MonkeyPatch):
    import duckdb

    def _boom(*args, **kwargs):
        raise duckdb.Error("boom")

    monkeypatch.setattr(duckdb, "connect", _boom)
    assert dsb.relation_count("select 1") is None


def test_run_oracle_records_rows_and_errors(tmp_path: Path):
    import duckdb

    root = tmp_path
    parquet = root / "t.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1), (2)) v(x)")
    con.execute(f"COPY t TO '{parquet}' (FORMAT PARQUET)")
    con.close()
    record = dsb.run_oracle(root, {"ok": "SELECT x FROM t", "bad": "SELECT * FROM no_such_table_xyz"}, ["t"])
    assert record["ok"]["rows"] == 2
    assert "rows_sha256" in record["ok"]
    assert "error" in record["bad"]
    assert sorted(dsb.load_oracle(root)["ok"]) == [(1,), (2,)]


def _write_dsb_source(src: Path) -> None:
    scripts = src / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "create_tables.sql").write_text("create table t\n(\n    a integer\n);")


def _fake_generate_queries(texts: dict[str, str]):
    def _fake(build, src, scale, seed, out, *args, **kwargs):
        out.mkdir(parents=True, exist_ok=True)
        return texts

    return _fake


def test_generate_rejects_scales_below_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="integer >= 1"):
        dsb.generate(0)


def test_generate_reuses_an_existing_dataset_and_refuses_to_overwrite_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    final = tmp_path / "dsb" / "sf1-dbgen-v1"
    final.mkdir(parents=True)
    (final / "t.parquet").write_bytes(b"x")
    assert dsb.generate(1) == final
    with pytest.raises(FileExistsError, match="immutable"):
        dsb.generate(1, force=True)


def test_generate_fails_when_dsdgen_writes_no_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    src = tmp_path / "src"
    _write_dsb_source(src)
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setattr(dsb, "fetch_source", lambda: src)
    monkeypatch.setattr(dsb, "build_tools", lambda _src: build)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="did not write"):
        dsb.generate(1)


def test_generate_converts_tables_queries_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    src = tmp_path / "src"
    _write_dsb_source(src)
    build = tmp_path / "build"
    build.mkdir()
    (build / "dsdgen").write_bytes(b"x")
    monkeypatch.setattr(dsb, "fetch_source", lambda: src)
    monkeypatch.setattr(dsb, "build_tools", lambda _src: build)

    def _fake_dsdgen(cmd, *args, **kwargs):
        raw = Path(cmd[cmd.index("-dir") + 1])
        (raw / "t.dat").write_bytes(b"1\n2\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_dsdgen)
    monkeypatch.setattr(dsb, "generate_queries", _fake_generate_queries({"agg_q001": "select 1\n"}))
    final = tmp_path / "dsb" / "sf1-dbgen-v1"
    final.mkdir(parents=True)  # empty: generate clears it instead of skipping
    out = dsb.generate(1, threads=1, oracle=False)
    assert out == final
    assert (out / "t.parquet").is_file()
    man = dsb.read_manifest(out)
    assert man is not None
    assert man["tables"]["t"]["rows"] == 2
    assert man["queries"] == {"agg_q001": {"sha256": hashlib.sha256(b"select 1\n").hexdigest(), "relations": 0}}
    assert man["relation_count_distribution"] == {"0": 1}
    assert man["oracle_file"] is None


def test_generate_records_oracle_answers_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    src = tmp_path / "src"
    _write_dsb_source(src)
    build = tmp_path / "build"
    build.mkdir()
    (build / "dsdgen").write_bytes(b"x")
    monkeypatch.setattr(dsb, "fetch_source", lambda: src)
    monkeypatch.setattr(dsb, "build_tools", lambda _src: build)

    def _fake_dsdgen(cmd, *args, **kwargs):
        raw = Path(cmd[cmd.index("-dir") + 1])
        (raw / "t.dat").write_bytes(b"1\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_dsdgen)
    monkeypatch.setattr(dsb, "generate_queries", _fake_generate_queries({"agg_q001": "select 1\n"}))

    def _fake_oracle(root, queries, tables, *args, **kwargs):
        dsb.save_oracle(root / dsb.ORACLE, {"agg_q001": [(1,)]})
        return {"agg_q001": {"rows": 1, "rows_sha256": "abc", "seconds": 0.1}}

    monkeypatch.setattr(dsb, "run_oracle", _fake_oracle)
    out = dsb.generate(1, threads=1, oracle=True)
    man = dsb.read_manifest(out)
    assert man is not None
    assert man["queries"]["agg_q001"]["oracle"] == {"rows": 1, "rows_sha256": "abc", "seconds": 0.1}
    assert man["oracle_file"] == dsb.ORACLE
    assert man["oracle_file_sha256"] == dsb.sha256_file(out / dsb.ORACLE)


def test_verify_reports_each_missing_file_and_checksum(tmp_path: Path):
    make_dataset(tmp_path)
    (tmp_path / "t.parquet").unlink()
    (tmp_path / "queries" / "agg_q013.sql").unlink()
    (tmp_path / dsb.ORACLE).write_bytes(b"changed")
    problems = dsb.verify(tmp_path)
    assert len(problems) == 3
    assert any("t.parquet" in p and "missing" in p for p in problems)
    assert any("query file is missing" in p for p in problems)
    assert any("sha256 differs" in p for p in problems)


def test_datasets_main_dsb_generates_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    from krishiv_bench import datasets

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    fake = tmp_path / "dsb" / "sf1-dbgen-v1"
    fake.mkdir(parents=True)
    seen: dict = {}

    def _fake_generate(scale, seed, threads, *args, **kwargs):
        seen["args"] = (scale, seed, threads, kwargs.get("oracle"))
        return fake

    monkeypatch.setattr("krishiv_bench.dsb.generate", _fake_generate)
    monkeypatch.setattr(datasets, "write_manifest", lambda d, stem: tmp_path / f"{stem}.sha256")
    monkeypatch.setattr("krishiv_bench.dsb.read_manifest", lambda d: {"relation_count_distribution": {0: 1}})
    datasets.main(["dsb", "--sf", "1"])
    assert seen["args"] == (1, 1, 1, True)
    assert "relations per query" in capsys.readouterr().out

    def _boom(*args, **kwargs):
        raise ValueError("bad scale")

    monkeypatch.setattr("krishiv_bench.dsb.generate", _boom)
    with pytest.raises(SystemExit, match="bad scale"):
        datasets.main(["dsb", "--sf", "1"])
    monkeypatch.setattr("krishiv_bench.dsb.verify", lambda d: [])
    datasets.main(["dsb-verify", str(fake)])
    assert "verified" in capsys.readouterr().out
    monkeypatch.setattr("krishiv_bench.dsb.verify", lambda d: ["boom"])
    with pytest.raises(SystemExit, match="boom"):
        datasets.main(["dsb-verify", str(fake)])


def test_run_suite_checks_dsb_oracle_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import json

    import duckdb

    from krishiv_bench.adapters.base import Engine

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    root = tmp_path / "dsb" / "sf1-dbgen-v1"
    (root / "queries").mkdir(parents=True)
    parquet = root / "t.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS x")
    con.execute(f"COPY t TO '{parquet}' (FORMAT PARQUET)")
    con.close()
    (root / "queries" / "q1.sql").write_text("SELECT x FROM t")
    (root / "queries" / "q2.sql").write_text("SELECT x FROM t")
    dsb.save_oracle(root / dsb.ORACLE, {"q1": [(1,)]})

    class _FakeEngine(Engine):
        def version(self) -> str:
            return "fake"

        def setup(self, dataset) -> None:
            pass

        def run(self, query):
            return [(1,)]

    monkeypatch.setattr(runner, "make_engine", lambda *a, **k: _FakeEngine())
    out = runner.run_suite("fake", "dsb", 1, iterations=1, warmup=0)
    rec = json.loads(out.read_text())
    assert rec["answers_available"] is True
    assert rec["queries"]["q1"]["answer_check"]["status"] == "pass"
    assert rec["queries"]["q2"]["answer_check"]["status"] == "not_checked"
    assert "no oracle answer" in rec["queries"]["q2"]["answer_check"]["detail"]
