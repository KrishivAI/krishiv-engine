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
