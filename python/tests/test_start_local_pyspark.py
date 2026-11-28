# SPDX-License-Identifier: Apache-2.0
"""``krishiv.start_local()`` with a stock ``pyspark-client`` (P3-T016); skipped without it."""

from __future__ import annotations

import os
import statistics
import sys
import time
from typing import Any

import pytest

import krishiv
import krishiv.sql.functions

pytest.importorskip("pyspark.sql.connect.session", reason="pyspark-client is not installed")
pytest.importorskip("pyarrow")
pytest.importorskip("grpc")


def _remote(url: str) -> Any:
    from pyspark.sql import SparkSession as PySparkSession

    return PySparkSession.builder.remote(url).create()


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


def test_a_client_without_the_token_is_refused(server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from pyspark.errors import PySparkException
    from pyspark.sql.connect.session import SparkSession as ConnectSession

    # ``use_ssl`` is off and the host is ``localhost``: the token is the only thing missing.
    monkeypatch.delenv("SPARK_CONNECT_AUTHENTICATE_TOKEN", raising=False)
    bare = f"sc://localhost:{server.port}"
    spark = ConnectSession.builder.remote(bare).create()
    try:
        with pytest.raises(PySparkException) as info:
            spark.range(1).collect()
        assert any(w in str(info.value).lower() for w in ("unauthenticated", "authentication"))
    finally:
        spark.stop()


def test_a_wrong_token_is_refused(server: Any) -> None:
    from pyspark.errors import PySparkException
    from pyspark.sql.connect.session import SparkSession as ConnectSession

    spark = ConnectSession.builder.remote(f"sc://localhost:{server.port}/;token={'0' * 64}").create()
    try:
        with pytest.raises(PySparkException) as info:
            spark.range(1).collect()
        assert any(w in str(info.value).lower() for w in ("unauthenticated", "authentication"))
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


def test_dataframe_window_functions_run_through_start_local() -> None:
    """QA2-2 / QA2-18: ``row_number().over(Window...)`` was refused as ``Expression.WINDOW(11)``."""
    from pyspark.sql import Window
    from pyspark.sql import functions as PF

    with krishiv.start_local() as server:
        spark = _remote(server.url)
        df = spark.createDataFrame([("a", 3), ("a", 5), ("b", 1), ("b", 9), ("b", 4)], "k string, v int")
        w = Window.partitionBy("k").orderBy(PF.desc("v"))
        top = df.withColumn("rk", PF.row_number().over(w)).filter("rk <= 1").orderBy("k").collect()
        assert [(r.k, r.v) for r in top] == [("a", 5), ("b", 9)]
        running = (
            df.withColumn(
                "s", PF.sum("v").over(Window.partitionBy("k").orderBy("v").rowsBetween(Window.unboundedPreceding, 0))
            )
            .orderBy("k", "v")
            .collect()
        )
        assert [r.s for r in running] == [3, 8, 1, 5, 14]
        spark.stop()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="UDF workers are Linux only")
def test_python_udfs_run_through_start_local() -> None:
    """QA2-4: ``F.udf`` and ``F.pandas_udf`` ran into 'no UDF worker pool' before."""
    from pyspark.sql import functions as PF

    with krishiv.start_local({"krishiv.python.worker.memoryLimit": "8g"}) as server:
        spark = _remote(server.url)
        df = spark.range(4).withColumnRenamed("id", "x")
        doubled = df.select(PF.udf(lambda x: (x or 0) * 2, "long")("x")).collect()
        assert [r[0] for r in doubled] == [0, 2, 4, 6]
        tripled = df.select(PF.pandas_udf(lambda x: x * 3, "long")("x")).collect()
        assert [r[0] for r in tripled] == [0, 3, 6, 9]
        spark.stop()


def test_guide_example_with_a_failing_query_exits_promptly(tmp_path: Any) -> None:
    """The guide's example (client stopped in `finally`) ends after a failed statement (CDX-2)."""
    import subprocess

    script = (
        "import krishiv\n"
        "from pyspark.sql import SparkSession\n"
        "with krishiv.start_local() as server:\n"
        "    spark = SparkSession.builder.remote(server.url).create()\n"
        "    try:\n"
        "        spark.range(3).show()\n"
        "        spark.sql(\"select cast('x' as double)\").collect()\n"
        "    finally:\n"
        "        spark.stop()\n"
    )
    path = tmp_path / "guide_example.py"
    path.write_text(script)
    done = subprocess.run([sys.executable, "-I", str(path)], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 1
    assert "CAST_INVALID_INPUT" in done.stderr
    assert "| id|" in done.stdout
