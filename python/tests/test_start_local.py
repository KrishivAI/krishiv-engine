# SPDX-License-Identifier: Apache-2.0
"""``krishiv.start_local()`` (P3-T016): the in-process Connect server for stock PySpark."""

from __future__ import annotations

import os
import socket
import statistics
import subprocess
import sys
import textwrap
import time
from typing import Any

import pytest

import krishiv
import krishiv.sql.functions


def _bindable(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _threads() -> int:
    return len(os.listdir("/proc/self/task"))


def _quiesced_threads() -> int:
    """The thread count once it has not changed for 300 ms (engine pool threads leave just after stop)."""
    last, since = _threads(), time.monotonic()
    while time.monotonic() - since < 0.3:
        time.sleep(0.02)
        now = _threads()
        if now != last:
            last, since = now, time.monotonic()
    return last


def _wait_threads(expected: int) -> None:
    deadline = time.monotonic() + 10
    while _threads() > expected and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _threads() == expected


def test_url_names_a_loopback_port_and_a_token() -> None:
    with krishiv.start_local() as server:
        assert server.running
        assert server.url == f"sc://localhost:{server.port}/;token={server.token}"
        assert str(server) == server.url
        assert len(server.token) == 64
        # Loopback only: the listener is on 127.0.0.1 and not on any other address.
        with socket.create_connection(("127.0.0.1", server.port), timeout=5):
            pass
        assert not _bindable(server.port)
    assert not server.running
    assert _bindable(server.port)


def test_every_start_has_its_own_token_and_port() -> None:
    with krishiv.start_local() as a, krishiv.start_local() as b:
        assert a.token != b.token
        assert a.port != b.port


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="counts threads through /proc")
def test_start_stop_100_times() -> None:
    krishiv.start_local().stop()  # lazily created process-wide threads exist after one start
    before = _quiesced_threads()
    ports = []
    for _ in range(100):
        server = krishiv.start_local()
        ports.append(server.port)
        server.stop()
        server.stop()  # a second stop does nothing
        assert not server.running
    assert all(_bindable(p) for p in ports)
    _wait_threads(before)


def test_interpreter_exit_stops_a_running_server() -> None:
    code = textwrap.dedent(
        """
        import krishiv
        server = krishiv.start_local()
        print(server.port, flush=True)
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert _bindable(int(out.stdout.split()[0]))


# --- stock pyspark-client ---------------------------------------------------------------------

pyspark_connect = pytest.importorskip("pyspark.sql.connect.session", reason="pyspark-client is not installed")
pytest.importorskip("pyarrow")
pytest.importorskip("grpc")


def _remote(url: str) -> Any:
    from pyspark.sql import SparkSession as PySparkSession

    return PySparkSession.builder.remote(url).getOrCreate()


@pytest.fixture
def server() -> Any:
    with krishiv.start_local() as s:
        yield s


def _rows(df: Any) -> list[tuple[Any, ...]]:
    return sorted((tuple(r) for r in df.collect()), key=repr)


def test_pyspark_client_runs_range_join_and_agg(server: Any) -> None:
    from pyspark.sql import functions as PF

    spark = _remote(server.url)
    try:
        embedded = krishiv.SparkSession.builder.create()
        assert _rows(spark.range(5)) == _rows(embedded.range(5))
        assert _rows(spark.range(2, 20, 3)) == _rows(embedded.range(2, 20, 3))

        def build(session: Any, f: Any) -> Any:
            left = session.range(40).select((f.col("id") % 4).alias("k"))
            right = session.range(20).select((f.col("id") % 4).alias("k"), f.col("id").alias("v"))
            return (
                left.join(right, "k").groupBy("k").agg(f.count(f.lit(1)).alias("n"), f.sum("v").alias("s")).orderBy("k")
            )

        remote_rows = [tuple(r) for r in build(spark, PF).collect()]
        assert remote_rows == [tuple(r) for r in build(embedded, krishiv.sql.functions).collect()]
        assert [r[1] for r in remote_rows] == [50, 50, 50, 50]
    finally:
        spark.stop()


@pytest.mark.xfail(
    reason="spark.sql sends a SqlCommand; Connect commands arrive with P3-T012 (56K00 until then)",
    strict=False,
)
def test_pyspark_client_runs_sql(server: Any) -> None:
    spark = _remote(server.url)
    try:
        embedded = krishiv.SparkSession.builder.create()
        sql = "SELECT id, id * 2 AS d FROM range(10) WHERE id % 3 = 0"
        assert _rows(spark.sql(sql)) == _rows(embedded.sql(sql))
        query = (
            "SELECT l.k, count(*) AS n, sum(r.v) AS s FROM "
            "(SELECT id % 4 AS k FROM range(40)) l JOIN "
            "(SELECT id % 4 AS k, id AS v FROM range(20)) r ON l.k = r.k GROUP BY l.k"
        )
        assert _rows(spark.sql(query)) == _rows(embedded.sql(query))
    finally:
        spark.stop()


def test_a_client_without_the_token_is_refused(server: Any) -> None:
    from pyspark.errors import PySparkException
    from pyspark.sql.connect.session import SparkSession as ConnectSession

    # ``use_ssl`` is off and the host is ``localhost``: the token is the only thing missing.
    bare = f"sc://localhost:{server.port}"
    spark = ConnectSession.builder.remote(bare).create()
    try:
        with pytest.raises(PySparkException) as info:
            spark.range(1).collect()
        assert "UNAUTHENTICATED" in str(info.value) or "authenticat" in str(info.value).lower()
    finally:
        spark.stop()


def test_a_wrong_token_is_refused(server: Any) -> None:
    from pyspark.errors import PySparkException
    from pyspark.sql.connect.session import SparkSession as ConnectSession

    spark = ConnectSession.builder.remote(f"sc://localhost:{server.port}/;token={'0' * 64}").create()
    try:
        with pytest.raises(PySparkException):
            spark.range(1).collect()
    finally:
        spark.stop()


# --- startup (measured, not gated) --------------------------------------------------------------


def test_startup_to_first_row_is_measured() -> None:
    """Time from ``start_local()`` to the first row at a stock client, over 20 starts.

    Measured, not gated: the median is printed (run with ``-s``) and only asserted when
    ``KRISHIV_ENFORCE_START_LOCAL=1`` (the benchmark box). The cold interpreter is excluded: one
    start and query run first.
    """
    from pyspark.sql.connect.session import SparkSession as ConnectSession

    def once() -> float:
        t0 = time.perf_counter()
        with krishiv.start_local() as s:
            spark = ConnectSession.builder.remote(s.url).create()
            try:
                assert spark.range(1).collect()[0][0] == 0
                return time.perf_counter() - t0
            finally:
                spark.stop()

    once()
    times = [once() for _ in range(20)]
    median = statistics.median(times)
    summary = f"median {median * 1000:.0f} ms, max {max(times) * 1000:.0f} ms"
    print(f"start_local to first row: {summary}, load {os.getloadavg()}")
    if os.environ.get("KRISHIV_ENFORCE_START_LOCAL") == "1":
        assert median < 1.0
