"""The Spark Connect adapter against a server on the host (Sail) needs no path rewrite and no container."""

from __future__ import annotations

from krishiv_bench.adapters.spark_connect import SparkConnectEngine


def test_host_server_has_no_path_rewrite_and_no_container():
    e = SparkConnectEngine(container=None, container_data_root=None)
    assert e.cdata == ""
    assert e.container is None


def test_container_default_still_rewrites():
    e = SparkConnectEngine()
    assert e.cdata == "/data"
    assert e.container == "krishiv-bench-spark"


class _Reader:
    def __init__(self, seen):
        self.seen = seen

    def parquet(self, path):
        self.seen.append(path)
        return self

    def createOrReplaceTempView(self, name):
        self.seen.append(f"view:{name}")


def _fake_session(monkeypatch, seen):
    import sys
    import types

    spark = types.SimpleNamespace(read=_Reader(seen))
    builder = types.SimpleNamespace(remote=lambda _r: types.SimpleNamespace(getOrCreate=lambda: spark))
    mod = types.SimpleNamespace(SparkSession=types.SimpleNamespace(builder=builder))
    monkeypatch.setitem(sys.modules, "pyspark.sql", mod)


def test_setup_reads_host_paths_from_a_host_server(monkeypatch, tmp_path):
    from krishiv_bench.adapters.base import Dataset

    seen: list[str] = []
    _fake_session(monkeypatch, seen)
    root = tmp_path / "tpch" / "sf1"
    e = SparkConnectEngine(container=None, container_data_root=None)
    e.setup(Dataset("tpch", 1, root, ["nation"]))
    assert seen == [f"{root}/nation.parquet", "view:nation"]


def test_setup_rewrites_to_the_container_mount(monkeypatch, tmp_path):
    from krishiv_bench.adapters.base import Dataset

    seen: list[str] = []
    _fake_session(monkeypatch, seen)
    root = tmp_path / "tpch" / "sf1"
    SparkConnectEngine(container=None).setup(Dataset("tpch", 1, root, ["nation"]))
    assert seen[0] == "/data/tpch/sf1/nation.parquet"


def test_timed_run_fetches_the_full_result_as_arrow_and_checksums_it(monkeypatch):
    import types

    import pyarrow as pa

    from krishiv_bench.adapters.base import Query
    from krishiv_bench.result_check import same

    table = pa.table({"k": [1, 2, 3], "m": [0.5, 1.5, 2.5]})
    asked: list[str] = []

    def sql(text):
        asked.append(text)
        return types.SimpleNamespace(toArrow=lambda: table)

    e = SparkConnectEngine(container=None, container_data_root=None)
    monkeypatch.setattr(e, "spark", types.SimpleNamespace(sql=sql))
    n, secs = e.timed_run(Query("q", "SELECT k, m FROM t", "tpch"))
    assert (n, asked) == (3, ["SELECT k, m FROM t"])
    assert secs >= 0
    first = e.result_checksum()
    assert first is not None
    assert e.result_checksum() is None  # taken once
    # The same result again gives the same checksum; a different one does not.
    e.timed_run(Query("q", "SELECT k, m FROM t", "tpch"))
    again = e.result_checksum()
    assert again is not None
    assert same(first, again)
    table = pa.table({"k": [1, 2, 3], "m": [0.5, 1.5, 2.6]})
    e.timed_run(Query("q", "SELECT k, m FROM t", "tpch"))
    other = e.result_checksum()
    assert other is not None
    assert not same(first, other)


def test_cli_passes_the_host_server_options(monkeypatch):
    from krishiv_bench import cli

    got = {}

    def fake(engine, suite, sf, it, wu, only, kw, *rest, **_kwargs):
        got.update(kw)
        from pathlib import Path

        return Path("/x")

    monkeypatch.setattr(cli, "run_suite", fake)
    cli.main(["run", "--engine", "spark-connect", "--spark-container", "", "--spark-data-root", ""])
    assert got["container"] is None
    assert got["container_data_root"] is None
