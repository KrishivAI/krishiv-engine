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
