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


def test_cli_passes_the_host_server_options(monkeypatch):
    from krishiv_bench import cli

    got = {}

    def fake(engine, suite, sf, it, wu, only, kw, *rest):
        got.update(kw)
        from pathlib import Path

        return Path("/x")

    monkeypatch.setattr(cli, "run_suite", fake)
    cli.main(["run", "--engine", "spark-connect", "--spark-container", "", "--spark-data-root", ""])
    assert got["container"] is None
    assert got["container_data_root"] is None
