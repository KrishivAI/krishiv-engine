# SPDX-License-Identifier: Apache-2.0
"""The krishiv-connect adapter: server command line, address parsing, registry (no server is started)."""

from __future__ import annotations

from typing import cast

import pytest

from krishiv_bench.adapters import ENGINES, make_engine
from krishiv_bench.adapters.base import Dataset
from krishiv_bench.adapters.krishiv_connect import KrishivConnectEngine, parse_address, server_binary
from krishiv_bench.clickbench_queries import SPARK_DIALECT


def test_parse_address_reads_the_startup_line_only():
    assert parse_address("krishiv server: Spark Connect on sc://127.0.0.1:41234") == "sc://127.0.0.1:41234"
    assert parse_address("krishiv server: admin on http://127.0.0.1:8080") is None


def test_command_binds_free_ports_and_passes_conf():
    e = KrishivConnectEngine(binary="/x/krishiv", conf=[("krishiv.a", "1"), ("b", "2")])
    assert e.command() == [
        "/x/krishiv", "server", "--port", "0", "--connect-port", "0", "--conf", "krishiv.a=1", "--conf", "b=2",
        "--conf", "krishiv.memory.limit=12g", "--conf", "krishiv.sql.session.timeZone=UTC",
    ]  # fmt: skip  # the runner's memory limit and time zone are the defaults (session_modes)


def test_host_server_has_no_path_rewrite_and_no_container():
    e = KrishivConnectEngine(binary="/x/krishiv")
    assert e.cdata == ""
    assert e.container is None


def test_registered_and_uses_the_spark_dialect():
    assert "krishiv-connect" in ENGINES
    assert isinstance(make_engine("krishiv-connect", binary="/x/krishiv"), KrishivConnectEngine)
    assert "krishiv-connect" in SPARK_DIALECT


def test_binary_env_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv("KRISHIV_BENCH_SERVER", "/env/krishiv")
    assert str(server_binary()) == "/env/krishiv"
    assert str(server_binary("/arg/krishiv")) == "/arg/krishiv"


def test_missing_binary_is_a_clear_error(tmp_path):
    e = KrishivConnectEngine(binary=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="KRISHIV_BENCH_SERVER"):
        e._start_server()


def _fake_server(tmp_path, body: str):
    p = tmp_path / "krishiv"
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(0o755)
    return p


def test_start_server_reads_the_address_and_teardown_stops_it(tmp_path):
    fake = _fake_server(tmp_path, "echo 'krishiv server: Spark Connect on sc://127.0.0.1:1'; sleep 60")
    e = KrishivConnectEngine(binary=fake, startup_timeout_s=10)
    e._start_server()
    assert e.remote == "sc://127.0.0.1:1"
    proc = e._proc
    assert proc is not None
    assert proc.poll() is None
    e.teardown()
    assert proc.poll() is not None


def test_server_that_exits_before_printing_an_address_is_an_error(tmp_path):
    e = KrishivConnectEngine(binary=_fake_server(tmp_path, "exit 3"), startup_timeout_s=10)
    with pytest.raises(RuntimeError, match="did not report"):
        e._start_server()


def test_server_that_hangs_silently_hits_the_startup_timeout(tmp_path):
    e = KrishivConnectEngine(binary=_fake_server(tmp_path, "sleep 60"), startup_timeout_s=0.5)
    with pytest.raises(RuntimeError, match="did not report"):
        e._start_server()
    assert e._proc is None


def test_binary_default_is_the_dist_build(monkeypatch, tmp_path):
    monkeypatch.delenv("KRISHIV_BENCH_SERVER", raising=False)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path))
    assert server_binary() == tmp_path / "dist" / "krishiv"


def test_start_server_twice_keeps_the_first_process(tmp_path):
    fake = _fake_server(tmp_path, "echo 'krishiv server: Spark Connect on sc://127.0.0.1:1'; sleep 60")
    e = KrishivConnectEngine(binary=fake, startup_timeout_s=10)
    try:
        e._start_server()
        first = e._proc
        e._start_server()
        assert e._proc is first
    finally:
        e.teardown()


def test_version_and_config_name_the_binary():
    e = KrishivConnectEngine(binary="/x/krishiv", conf=[("a", "1")])
    assert "krishiv" in e.version()
    assert e.config()["binary"] == "/x/krishiv"
    assert e.config()["conf"] == [("a", "1"), ("krishiv.memory.limit", "12g"), ("krishiv.sql.session.timeZone", "UTC")]


def test_setup_failure_stops_the_server(tmp_path, monkeypatch):
    from krishiv_bench.adapters.spark_connect import SparkConnectEngine

    def boom(self, dataset):
        raise ValueError("no session")

    monkeypatch.setattr(SparkConnectEngine, "setup", boom)
    fake = _fake_server(tmp_path, "echo 'krishiv server: Spark Connect on sc://127.0.0.1:1'; sleep 60")
    e = KrishivConnectEngine(binary=fake, startup_timeout_s=10)
    with pytest.raises(ValueError, match="no session"):
        e.setup(cast(Dataset, None))
    assert e._proc is None


def test_teardown_stops_the_server_even_when_the_session_stop_raises(tmp_path, monkeypatch):
    from krishiv_bench.adapters.spark_connect import SparkConnectEngine

    def boom(self):
        raise OSError("stop failed")

    monkeypatch.setattr(SparkConnectEngine, "teardown", boom)
    fake = _fake_server(tmp_path, "echo 'krishiv server: Spark Connect on sc://127.0.0.1:1'; sleep 60")
    e = KrishivConnectEngine(binary=fake, startup_timeout_s=10)
    e._start_server()
    proc = e._proc
    assert proc is not None
    e.begin_measure()
    with pytest.raises(OSError, match="stop failed"):
        e.teardown()
    assert proc.wait(timeout=5) is not None
    assert e._proc is None
    assert e._tree is None


def test_setup_success_leaves_the_server_running(tmp_path, monkeypatch):
    from krishiv_bench.adapters.spark_connect import SparkConnectEngine

    monkeypatch.setattr(SparkConnectEngine, "setup", lambda self, dataset: None)
    fake = _fake_server(tmp_path, "echo 'krishiv server: Spark Connect on sc://127.0.0.1:1'; sleep 60")
    e = KrishivConnectEngine(binary=fake, startup_timeout_s=10)
    try:
        e.setup(cast(Dataset, None))
        assert e._proc is not None
    finally:
        e.teardown()


def test_measure_window_covers_the_server_process(tmp_path):
    fake = _fake_server(tmp_path, "echo 'krishiv server: Spark Connect on sc://127.0.0.1:1'; sleep 60")
    e = KrishivConnectEngine(binary=fake, startup_timeout_s=10)
    try:
        e._start_server()
        e.begin_measure()
        assert e._tree is not None
        m = e.end_measure()
        assert m is not None
        assert e._tree is None
    finally:
        e.teardown()


def test_teardown_kills_a_server_that_ignores_sigterm(tmp_path, monkeypatch):
    fake = _fake_server(
        tmp_path, "trap '' TERM\necho 'krishiv server: Spark Connect on sc://127.0.0.1:1'\nwhile true; do sleep 1; done"
    )
    e = KrishivConnectEngine(binary=fake, startup_timeout_s=10)
    e._start_server()
    proc = e._proc
    assert proc is not None
    import krishiv_bench.adapters.krishiv_connect as mod

    real_wait = proc.wait
    calls = []

    def fast_wait(timeout=None):
        calls.append(timeout)
        return real_wait(timeout=0.3)

    monkeypatch.setattr(proc, "wait", fast_wait)
    e.teardown()
    assert calls
    assert real_wait(timeout=5) is not None
    assert mod is not None
