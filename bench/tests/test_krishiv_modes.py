# SPDX-License-Identifier: Apache-2.0
"""The krishiv-python and krishiv-flightsql adapters and the `run` options of the mode comparison (no engine starts)."""

from __future__ import annotations

import pytest

from krishiv_bench import cli, runner
from krishiv_bench.adapters import ENGINES, make_engine
from krishiv_bench.adapters.krishiv_flightsql import KrishivFlightSqlEngine
from krishiv_bench.adapters.krishiv_python import KrishivPythonEngine
from krishiv_bench.clickbench_queries import SPARK_DIALECT


@pytest.mark.parametrize(
    ("name", "cls"), [("krishiv-python", KrishivPythonEngine), ("krishiv-flightsql", KrishivFlightSqlEngine)]
)
def test_registered_and_uses_the_spark_dialect(name, cls):
    assert name in ENGINES
    kw = {"binary": "/x/krishiv"} if name == "krishiv-flightsql" else {}
    assert isinstance(make_engine(name, **kw), cls)
    assert name in SPARK_DIALECT


def test_flightsql_uses_the_connect_port_and_the_same_server_command():
    e = KrishivFlightSqlEngine(binary="/x/krishiv", conf=[("krishiv.memory.limit", "12g")])
    e.remote = "sc://127.0.0.1:41234"
    assert e._uri() == "grpc://127.0.0.1:41234"
    e.remote = "sc://127.0.0.1:41234/;token=t"
    assert e._uri() == "grpc://127.0.0.1:41234"
    assert e.command() == [
        "/x/krishiv", "server", "--port", "0", "--connect-port", "0", "--conf", "krishiv.memory.limit=12g",
    ]  # fmt: skip


def test_python_engine_keeps_its_conf_and_measures_nothing_before_a_window():
    e = KrishivPythonEngine(conf=[("krishiv.memory.limit", "12g")])
    assert e.conf == [("krishiv.memory.limit", "12g")]
    assert e.container is None
    assert e.end_measure().peak_rss_bytes is None


def test_krishiv_conf_reaches_every_krishiv_mode(monkeypatch):
    seen: dict[str, dict] = {}

    def fake_run_suite(engine, suite, sf, iterations, warmup, only, kw, *rest, **kwrest):
        seen[engine] = kw
        return "result.json"

    monkeypatch.setattr(cli, "run_suite", fake_run_suite)
    for engine in ("krishiv", "krishiv-connect", "krishiv-flightsql", "krishiv-python"):
        cli.main(["run", "--engine", engine, "--suite", "clickbench", "--krishiv-conf", "krishiv.memory.limit=12g"])
        assert seen[engine]["conf"] == [("krishiv.memory.limit", "12g")]
    cli.main(["run", "--engine", "krishiv-flightsql", "--runner", "/x/krishiv"])
    assert seen["krishiv-flightsql"]["binary"] == "/x/krishiv"


def test_krishiv_conf_needs_key_value():
    with pytest.raises(SystemExit):
        cli.main(["run", "--engine", "krishiv", "--krishiv-conf", "novalue"])


def test_clickbench_dataset_has_no_scale_factor(monkeypatch, tmp_path):
    (tmp_path / "clickbench").mkdir()
    monkeypatch.setattr(runner, "data_root", lambda: tmp_path)
    with pytest.raises(SystemExit, match=r"no dataset at .*clickbench;"):
        runner.run_suite("krishiv", "clickbench", 1, 1, 0, None, {})


def test_query_log_is_written_only_when_asked(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from krishiv_bench.adapters import query_log

    asked: list[str] = []

    def fetch(sql: str):
        asked.append(sql)
        return pa.table({"query_id": ["a"], "wall_ns": [5]})

    monkeypatch.delenv("KRISHIV_BENCH_QUERY_LOG", raising=False)
    assert query_log.dump("krishiv-python", fetch) is None
    assert asked == []
    monkeypatch.setenv("KRISHIV_BENCH_QUERY_LOG", str(tmp_path / "log"))
    path = query_log.dump("krishiv-python", fetch)
    assert path is not None
    assert path.name.startswith("krishiv-python-")
    assert pq.read_table(path).column("wall_ns").to_pylist() == [5]
    assert "system.queries" in asked[0]

    def broken(sql: str):
        raise RuntimeError("server gone")

    assert query_log.dump("krishiv-connect", broken) is None


def test_timed_run_times_the_arrow_path_and_checks_one_untimed_run(monkeypatch, tmp_path):
    import json

    from krishiv_bench.adapters.base import Engine, Query

    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    d = tmp_path / "tpch" / "sf1"
    d.mkdir(parents=True)
    (d / "t.parquet").write_bytes(b"")
    calls: list[str] = []

    class _Fake(Engine):
        def version(self):
            return "fake"

        def setup(self, dataset):
            pass

        def run(self, query):
            calls.append("run")
            return [(1,), (2,)]

        def timed_run(self, query):
            calls.append("timed")
            return 2, 0.25 + 0.5 * calls.count("timed")

    monkeypatch.setattr(runner, "make_engine", lambda *a, **k: _Fake())
    monkeypatch.setattr(runner, "load_queries", lambda *a, **k: [Query("q01", "select 1", "tpch")])
    monkeypatch.setattr(runner, "dataset_provenance", lambda *a, **k: {})
    out = runner.run_suite("fake", "tpch", 1, iterations=2, warmup=1, min_mem_avail=None, timed_run=True)
    rec = json.loads(out.read_text())
    q = rec["queries"]["q01"]
    assert calls == ["timed", "timed", "timed", "run"]
    assert q["warmup_times_s"] == [0.75]
    assert q["times_s"] == [1.25, 1.75]  # the engine's seconds, not the harness's clock
    assert q["rows_returned"] == 2
    assert rec["timed_run"] is True
    assert rec["setup_s"] >= 0
