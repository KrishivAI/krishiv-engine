# SPDX-License-Identifier: Apache-2.0
"""The krishiv-connect adapter: server command line, address parsing, registry (no server is started)."""

from __future__ import annotations

import pytest

from krishiv_bench.adapters import ENGINES, make_engine
from krishiv_bench.adapters.krishiv_connect import KrishivConnectEngine, parse_address, server_binary
from krishiv_bench.clickbench_queries import SPARK_DIALECT


def test_parse_address_reads_the_startup_line_only():
    assert parse_address("krishiv server: Spark Connect on sc://127.0.0.1:41234") == "sc://127.0.0.1:41234"
    assert parse_address("krishiv server: admin on http://127.0.0.1:8080") is None


def test_command_binds_free_ports_and_passes_conf():
    e = KrishivConnectEngine(binary="/x/krishiv", conf=[("krishiv.a", "1"), ("b", "2")])
    assert e.command() == [
        "/x/krishiv", "server", "--port", "0", "--connect-port", "0", "--conf", "krishiv.a=1", "--conf", "b=2",
    ]  # fmt: skip


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
    assert proc is not None and proc.poll() is None
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
