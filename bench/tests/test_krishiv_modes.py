# SPDX-License-Identifier: Apache-2.0
"""The mode comparison: krishiv-python and krishiv-flightsql, the session modes' shared defaults, the query log and
the `run --timed-run` path (no engine or server starts)."""

from __future__ import annotations

import json
import sys
import threading
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from krishiv_bench import cli, runner
from krishiv_bench.adapters import ENGINES, make_engine, query_log, session_modes
from krishiv_bench.adapters.base import Dataset, Engine, Query
from krishiv_bench.adapters.krishiv_connect import KrishivConnectEngine
from krishiv_bench.adapters.krishiv_flightsql import KrishivFlightSqlEngine
from krishiv_bench.adapters.krishiv_python import KrishivPythonEngine
from krishiv_bench.clickbench_queries import SPARK_DIALECT

DEFAULTS = [("krishiv.memory.limit", "12g"), ("krishiv.sql.session.timeZone", "UTC")]


@pytest.mark.parametrize(
    ("name", "cls"), [("krishiv-python", KrishivPythonEngine), ("krishiv-flightsql", KrishivFlightSqlEngine)]
)
def test_registered_and_uses_the_spark_dialect(name, cls):
    assert name in ENGINES
    kw = {"binary": "/x/krishiv"} if name == "krishiv-flightsql" else {}
    assert isinstance(make_engine(name, **kw), cls)
    assert name in SPARK_DIALECT


def test_flightsql_uses_the_connect_port_and_the_same_server_command():
    e = KrishivFlightSqlEngine(binary="/x/krishiv")
    e.remote = "sc://127.0.0.1:41234"
    assert e._uri() == "grpc://127.0.0.1:41234"
    e.remote = "sc://127.0.0.1:41234/;token=t"
    assert e._uri() == "grpc://127.0.0.1:41234"
    assert e.command() == [
        "/x/krishiv", "server", "--port", "0", "--connect-port", "0",
        "--conf", "krishiv.memory.limit=12g", "--conf", "krishiv.sql.session.timeZone=UTC",
    ]  # fmt: skip


# ---- the session modes start from the runner's settings


@pytest.mark.parametrize("name", ["krishiv-python", "krishiv-connect", "krishiv-flightsql"])
def test_session_modes_default_to_the_runners_settings_and_record_them(name):
    kw = {} if name == "krishiv-python" else {"binary": "/x/krishiv"}
    e = make_engine(name, **kw)
    assert isinstance(e, KrishivPythonEngine | KrishivConnectEngine)
    assert e.conf == DEFAULTS
    eff = session_modes.effective(e.conf)
    assert eff == {"memory_limit": "12g", "time_zone": "UTC", "threads": session_modes.usable_cores()}
    if name != "krishiv-python":  # krishiv-python's config() imports the wheel
        assert e.config()["effective"] == eff


def test_an_explicit_setting_wins_over_the_default():
    conf = session_modes.with_runner_defaults([("krishiv.memory.limit", "20g"), ("spark.sql.session.timeZone", "Z")])
    assert conf == [("krishiv.memory.limit", "20g"), ("spark.sql.session.timeZone", "Z")]
    assert session_modes.effective(conf)["time_zone"] == "Z"


def test_python_engine_measures_nothing_before_a_window():
    e = KrishivPythonEngine()
    assert e.container is None
    assert e.end_measure().peak_rss_bytes is None


# ---- cli


def _capture(monkeypatch) -> dict[str, dict]:
    seen: dict[str, dict] = {}

    def fake_run_suite(engine, suite, sf, iterations, warmup, only, kw, *rest, **kwrest):
        seen[engine] = {**kw, "_timed_run": kwrest.get("timed_run")}
        return "result.json"

    monkeypatch.setattr(cli, "run_suite", fake_run_suite)
    return seen


def test_krishiv_conf_and_timed_run_reach_every_krishiv_mode(monkeypatch):
    seen = _capture(monkeypatch)
    for engine in ("krishiv", "krishiv-connect", "krishiv-flightsql", "krishiv-python"):
        cli.main(["run", "--engine", engine, "--suite", "clickbench", "--timed-run", "--krishiv-conf", "a.b=1"])
        assert seen[engine]["conf"] == [("a.b", "1")]
        assert seen[engine]["_timed_run"] is True
    cli.main(["run", "--engine", "krishiv-flightsql", "--runner", "/x/krishiv"])
    assert seen["krishiv-flightsql"]["binary"] == "/x/krishiv"


def test_krishiv_conf_needs_key_value():
    with pytest.raises(SystemExit):
        cli.main(["run", "--engine", "krishiv", "--krishiv-conf", "novalue"])


@pytest.mark.parametrize("name", ["krishiv-python", "krishiv-connect", "krishiv-flightsql"])
def test_session_modes_refuse_a_thread_count_they_cannot_apply(monkeypatch, name):
    seen = _capture(monkeypatch)
    monkeypatch.setattr(session_modes, "usable_cores", lambda: 12)
    with pytest.raises(SystemExit):
        cli.main(["run", "--engine", name, "--duckdb-threads", "8"])
    cli.main(["run", "--engine", name, "--duckdb-threads", "12"])
    assert name in seen
    cli.main(["run", "--engine", "krishiv", "--duckdb-threads", "8"])  # the runner takes any count
    assert seen["krishiv"]["threads"] == 8


def test_clickbench_dataset_has_no_scale_factor(monkeypatch, tmp_path):
    (tmp_path / "clickbench").mkdir()
    monkeypatch.setattr(runner, "data_root", lambda: tmp_path)
    with pytest.raises(SystemExit, match=r"no dataset at .*clickbench;"):
        runner.run_suite("krishiv", "clickbench", 1, 1, 0, None, {})


# ---- the query log


def test_query_log_writes_only_new_rows_and_only_when_asked(monkeypatch, tmp_path):
    rows = {"query_id": ["a"], "wall_ns": [5]}
    asked: list[str] = []

    def fetch(sql: str):
        asked.append(sql)
        return pa.table(rows)

    log = query_log.QueryLog("krishiv-python")
    monkeypatch.delenv("KRISHIV_BENCH_QUERY_LOG", raising=False)
    assert log.flush(fetch) is None
    assert asked == []
    monkeypatch.setenv("KRISHIV_BENCH_QUERY_LOG", str(tmp_path / "log"))
    first = log.flush(fetch)
    assert first is not None
    assert first.name.startswith("krishiv-python-")
    assert pq.read_table(first).column("wall_ns").to_pylist() == [5]
    assert "system.queries" in asked[0]
    assert log.flush(fetch) is None  # nothing new
    rows = {"query_id": ["a", "b"], "wall_ns": [5, 7]}
    second = log.flush(fetch)
    assert second is not None
    assert pq.read_table(second).column("query_id").to_pylist() == ["b"]

    def broken(sql: str):
        raise RuntimeError("server gone")

    assert log.flush(broken) is None


def test_mark_sql_keeps_only_safe_characters():
    assert query_log.mark_sql("q01 run 2 attempt 0") == "SELECT 'q01 run 2 attempt 0' AS krishiv_bench_mark"
    assert query_log.mark_sql("q'; DROP x") == "SELECT 'q DROP x' AS krishiv_bench_mark"


# ---- krishiv-flightsql against a fake ADBC driver


class _FakeServer:
    def __init__(self):
        self.executed: list[str] = []
        self.connections: list[_FakeConn] = []
        self.blocking = threading.Event()  # a `SELECT block` statement is inside execute
        self.unblock = threading.Event()  # ends it without a cancel
        self.cancel_fails = False
        self.blocked: list[_FakeCursor] = []  # cursors inside a `SELECT block`, in order

    def connect(self, uri: str, db_kwargs: dict | None = None, autocommit: bool = False):
        assert uri == "grpc://127.0.0.1:1"
        assert autocommit
        # Without the cookie middleware every statement gets a new server session and the views are gone.
        assert (db_kwargs or {}).get("adbc.flight.sql.rpc.with_cookie_middleware") == "true"
        c = _FakeConn(self)
        self.connections.append(c)
        return c


class _FakeConn:
    def __init__(self, server: _FakeServer):
        self.server, self.closed = server, False

    def cursor(self):
        return _FakeCursor(self)

    def close(self) -> None:
        self.closed = True


class _FakeCursor:
    def __init__(self, conn: _FakeConn):
        self.conn = conn
        self.cancelled = threading.Event()
        self.release = threading.Event()  # ends this one statement without a cancel

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str) -> None:
        if self.conn.closed:
            raise RuntimeError("connection closed")
        server = self.conn.server
        server.executed.append(sql)
        if "block" in sql:
            server.blocked.append(self)
            server.blocking.set()
            while not (self.cancelled.wait(0.01) or server.unblock.is_set() or self.release.is_set()):
                pass
            raise RuntimeError("cancelled" if self.cancelled.is_set() else "ended")

    def fetch_arrow_table(self):
        return pa.table({"x": [1, 2, 3]})

    def fetchall(self):
        return [(1,), (2,), (3,)]

    def adbc_cancel(self) -> None:
        if self.conn.server.cancel_fails:
            raise RuntimeError("cancel not supported")
        self.cancelled.set()


@pytest.fixture
def flight(monkeypatch, tmp_path):
    server = _FakeServer()
    pkg = types.ModuleType("adbc_driver_flightsql")
    dbapi = types.ModuleType("adbc_driver_flightsql.dbapi")
    dbapi.connect = server.connect  # ty: ignore[unresolved-attribute]
    pkg.dbapi = dbapi  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "adbc_driver_flightsql", pkg)
    monkeypatch.setitem(sys.modules, "adbc_driver_flightsql.dbapi", dbapi)
    e = KrishivFlightSqlEngine(binary="/x/krishiv")
    monkeypatch.setattr(e, "_start_server", lambda: setattr(e, "remote", "sc://127.0.0.1:1"))
    e.setup(Dataset("tpch", 1, tmp_path, ["lineitem", "orders"]))
    return e, server, tmp_path


def _views(root) -> list[str]:
    return [
        f"CREATE TABLE IF NOT EXISTS lineitem USING parquet LOCATION '{root / 'lineitem.parquet'}'",
        f"CREATE TABLE IF NOT EXISTS orders USING parquet LOCATION '{root / 'orders.parquet'}'",
    ]


def _start_blocked(e, server) -> tuple[threading.Thread, dict]:
    done: dict = {}
    server.blocking.clear()

    def worker():
        try:
            e.timed_run(Query("q99", "SELECT block", "tpch"))
        except Exception as ex:
            done["error"] = str(ex)

    t = threading.Thread(target=worker)
    t.start()
    assert server.blocking.wait(5)
    return t, done


def test_flightsql_registers_views_and_times_the_arrow_fetch(flight):
    e, server, root = flight
    assert server.executed == _views(root)
    n, secs = e.timed_run(Query("q01", "SELECT 1", "tpch"))
    assert n == 3
    assert secs >= 0
    assert e._pending is not None
    e.release_result()
    assert e._pending is None
    assert e.run(Query("q01", "SELECT 1", "tpch")) == [(1,), (2,), (3,)]


def test_flightsql_cancel_keeps_the_connection_and_its_views(flight):
    e, server, _ = flight
    t, done = _start_blocked(e, server)
    e.cancel()
    t.join(5)
    assert done["error"] == "cancelled"
    assert server.connections[0].closed is False
    assert e.timed_run(Query("q01", "SELECT 1", "tpch"))[0] == 3  # the next query still runs
    assert len(server.connections) == 1


def test_flightsql_reconnects_and_registers_again_when_cancel_fails(flight):
    e, server, root = flight
    server.cancel_fails = True
    t, done = _start_blocked(e, server)
    e.cancel()
    first = server.connections[0]
    assert first.closed is False  # never closed under the thread inside its cursor
    server.unblock.set()  # the statement ends by itself (the harness waits up to 30 s for it)
    t.join(5)
    assert done["error"] == "ended"
    server.executed.clear()
    assert e.timed_run(Query("q01", "SELECT 1", "tpch"))[0] == 3
    assert first.closed is True
    assert len(server.connections) == 2
    assert server.executed == [*_views(root), "SELECT 1"]


def test_flightsql_never_closes_a_connection_while_a_statement_runs_on_it(flight):
    e, server, root = flight
    server.cancel_fails = True
    t, done = _start_blocked(e, server)
    e.cancel()  # fails: the statement keeps running past the harness's 30 s wait
    first = server.connections[0]
    server.executed.clear()
    assert e.timed_run(Query("q01", "SELECT 1", "tpch"))[0] == 3  # the next query, on a new connection
    assert len(server.connections) == 2
    assert server.executed == [*_views(root), "SELECT 1"]
    assert first.closed is False  # the old statement is still inside it
    server.blocked[0].release.set()
    t.join(5)
    assert done["error"] == "ended"
    assert first.closed is True  # closed as its last statement returned
    assert server.connections[1].closed is False


def test_flightsql_cancel_reaches_a_statement_that_overlaps_a_late_one(flight):
    e, server, _ = flight
    server.cancel_fails = True
    t1, done1 = _start_blocked(e, server)
    e.cancel()  # fails: the first statement is left running
    server.cancel_fails = False
    t2, done2 = _start_blocked(e, server)  # the next query, on a new connection
    server.blocked[0].release.set()  # the first one ends now; it must not clear the second's cursor
    t1.join(5)
    assert done1["error"] == "ended"
    e.cancel()
    t2.join(5)
    assert done2["error"] == "cancelled"
    assert server.connections[1].closed is False


def test_flightsql_teardown_never_closes_a_connection_with_a_statement_inside(flight, monkeypatch):
    e, server, _ = flight
    server.cancel_fails = True
    t, done = _start_blocked(e, server)
    e.cancel()  # fails: the statement is left running on the first connection
    e.timed_run(Query("q01", "SELECT 1", "tpch"))  # the second connection
    monkeypatch.setattr(KrishivConnectEngine, "teardown", lambda self: None)
    e.teardown()
    first, second = server.connections
    assert first.closed is False  # its statement is still inside
    assert second.closed is True  # idle: closed at once
    server.blocked[0].release.set()  # the server's stop ends it
    t.join(5)
    assert done["error"] == "ended"
    assert first.closed is True  # closed as its last statement returned


def test_flightsql_teardown_leaves_the_current_connection_open_while_its_statement_runs(flight, monkeypatch):
    e, server, _ = flight
    t, _ = _start_blocked(e, server)  # a timed-out statement that did not stop in the 30 s wait
    monkeypatch.setattr(KrishivConnectEngine, "teardown", lambda self: None)
    e.teardown()
    assert server.connections[0].closed is False
    server.blocked[0].release.set()
    t.join(5)
    assert server.connections[0].closed is True


def test_flightsql_a_failed_cancel_on_a_retired_connection_keeps_the_current_one(flight):
    e, server, _ = flight
    server.cancel_fails = True
    t1, _ = _start_blocked(e, server)
    e.cancel()  # fails: the first connection is retired at the next statement
    t2, done2 = _start_blocked(e, server)  # runs on the second connection
    assert len(server.connections) == 2
    server.blocked[1].cancelled.set()  # the second statement ends as if its cancel worked
    t2.join(5)
    assert done2["error"] == "cancelled"
    e.cancel()  # only the first statement is left; its cancel fails again
    assert e.timed_run(Query("q01", "SELECT 1", "tpch"))[0] == 3
    assert len(server.connections) == 2  # no reconnect: the current connection is fine
    server.blocked[0].release.set()
    t1.join(5)


def test_flightsql_marks_and_flushes_through_its_connection(flight, monkeypatch, tmp_path):
    e, server, _ = flight
    monkeypatch.setenv("KRISHIV_BENCH_QUERY_LOG", str(tmp_path / "log"))
    e.mark("q01 run 0 attempt 0")
    assert server.executed[-1] == "SELECT 'q01 run 0 attempt 0' AS krishiv_bench_mark"
    monkeypatch.setattr(e, "_proc", object())
    monkeypatch.setattr(_FakeCursor, "fetch_arrow_table", lambda self: pa.table({"query_id": ["a"]}))
    e.flush_query_log()
    assert "system.queries" in server.executed[-1]
    assert len(list((tmp_path / "log").glob("krishiv-flightsql-*.parquet"))) == 1


class _SqlSpark:
    def __init__(self):
        self.sql_text: list[str] = []

    def sql(self, text: str):
        self.sql_text.append(text)


def test_krishiv_python_registers_catalog_tables_with_sql(monkeypatch, tmp_path):
    spark = _SqlSpark()
    builder = types.SimpleNamespace(config=lambda k, v: builder, getOrCreate=lambda: spark)
    fake = types.ModuleType("krishiv")
    fake.SparkSession = types.SimpleNamespace(builder=builder)  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "krishiv", fake)
    e = KrishivPythonEngine()
    e.setup(Dataset("tpch", 1, tmp_path, ["lineitem", "orders"]))
    assert spark.sql_text == _views(tmp_path)


def test_krishiv_connect_registers_catalog_tables_with_sql():
    e = KrishivConnectEngine(binary="/x/krishiv")
    spark = _SqlSpark()
    e.spark = spark  # ty: ignore[invalid-assignment]
    e.register_view("lineitem", "/d/sf10/lineitem.parquet")
    # The server reads no temporary view over Spark Connect (session_modes docstring).
    assert spark.sql_text == ["CREATE TABLE IF NOT EXISTS lineitem USING parquet LOCATION '/d/sf10/lineitem.parquet'"]


@pytest.mark.parametrize(("table", "path"), [("li`x", "/d/a.parquet"), ("t", "/d/o'x.parquet"), ("t", "/d/a\\b")])
def test_table_sql_refuses_what_would_need_quoting(table, path):
    with pytest.raises(ValueError, match="needs quoting"):
        session_modes.table_sql(table, path)


def test_the_query_log_reads_rows_in_submission_order():
    assert query_log.SQL.endswith("FROM system.queries ORDER BY submitted_at")


def test_usable_cores_applies_the_tightest_cgroup_quota(tmp_path):
    (tmp_path / "self").write_text("0::/a/b\n")
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (tmp_path / "a" / "cpu.max").write_text("250000 100000\n")  # 2.5 cores: 2, as available_parallelism
    (tmp_path / "a" / "b" / "cpu.max").write_text("max 100000\n")
    assert session_modes.cgroup_cpu_limit(str(tmp_path), str(tmp_path / "self")) == 2
    (tmp_path / "a" / "b" / "cpu.max").write_text("50000 100000\n")
    assert session_modes.cgroup_cpu_limit(str(tmp_path), str(tmp_path / "self")) == 1
    (tmp_path / "a" / "cpu.max").write_text("max 100000\n")
    (tmp_path / "a" / "b" / "cpu.max").write_text("max 100000\n")
    assert session_modes.cgroup_cpu_limit(str(tmp_path), str(tmp_path / "self")) is None
    assert session_modes.cgroup_cpu_limit(str(tmp_path), str(tmp_path / "missing")) is None


# ---- run_suite --timed-run


def _suite(monkeypatch, tmp_path, engine: Engine, **kw):
    monkeypatch.setenv("KRISHIV_DATA", str(tmp_path))
    d = tmp_path / "tpch" / "sf1"
    d.mkdir(parents=True)
    (d / "t.parquet").write_bytes(b"")
    monkeypatch.setattr(runner, "make_engine", lambda *a, **k: engine)
    monkeypatch.setattr(runner, "load_queries", lambda *a, **k: [Query("q01", "select 1", "tpch")])
    monkeypatch.setattr(runner, "dataset_provenance", lambda *a, **k: {})
    out = runner.run_suite("fake", "tpch", 1, min_mem_avail=None, timed_run=True, **kw)
    return json.loads(out.read_text())


class _Fake(Engine):
    def __init__(self, block_answer: bool = False):
        self.calls: list[str] = []
        self.block_answer = block_answer
        self.release = threading.Event()
        self.mark_fails = False

    def version(self):
        return "fake"

    def setup(self, dataset):
        pass

    def run(self, query):
        self.calls.append("run")
        if self.block_answer:
            self.release.wait(5)
        return [(1,), (2,)]

    def timed_run(self, query):
        self.calls.append("timed")
        self._pending = "result"
        return 2, 0.25 + 0.5 * self.calls.count("timed")

    def release_result(self):
        self.calls.append("release")
        super().release_result()

    def mark(self, label):
        self.calls.append(f"mark {label}")
        if self.mark_fails:
            raise RuntimeError("marker refused")

    def flush_query_log(self):
        self.calls.append("flush")

    def cancel(self):
        self.calls.append("cancel")
        self.release.set()


def test_timed_run_times_the_arrow_path_and_checks_one_untimed_run(monkeypatch, tmp_path):
    eng = _Fake()
    rec = _suite(monkeypatch, tmp_path, eng, iterations=2, warmup=1)
    q = rec["queries"]["q01"]
    assert eng.calls == [
        "flush",
        "mark q01 warmup 0 attempt 0", "timed", "release",
        "mark q01 run 1 attempt 0", "timed", "release",
        "mark q01 run 2 attempt 0", "timed", "release",
        "mark q01 answer", "run",
    ]  # fmt: skip
    assert q["warmup_times_s"] == [0.75]
    assert q["times_s"] == [1.25, 1.75]  # the engine's seconds, not the harness's clock
    assert q["rows_returned"] == 2
    assert rec["timed_run"] is True
    assert rec["setup_s"] >= 0


def test_a_slow_answer_run_is_cancelled_and_the_timings_are_kept(monkeypatch, tmp_path):
    eng = _Fake(block_answer=True)
    rec = _suite(monkeypatch, tmp_path, eng, iterations=2, warmup=0, timeout_s=0.2)
    q = rec["queries"]["q01"]
    assert "cancel" in eng.calls
    assert q["times_s"] == [0.75, 1.25]
    assert q["rows_returned"] == 2  # from the timed runs
    assert q["answer_check"]["status"] == "error"
    assert "TimeoutError" in q["answer_check"]["detail"]
    assert q["rows_stored"] is False


def test_a_failed_marker_never_fails_the_run_it_labels(monkeypatch, tmp_path):
    eng = _Fake()
    eng.mark_fails = True
    rec = _suite(monkeypatch, tmp_path, eng, iterations=2, warmup=0)
    q = rec["queries"]["q01"]
    assert "error" not in q
    assert q["times_s"] == [0.75, 1.25]
    assert q["answer_check"]["status"] != "error"
