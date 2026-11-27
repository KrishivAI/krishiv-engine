# SPDX-License-Identifier: Apache-2.0
"""``DataFrame.plot`` (Spark 4.0+) from a stock ``pyspark-client`` against ``krishiv.start_local()``.

The plotting code runs in the client and reads ``spark.sql.pyspark.plotting.max_rows`` from the server;
the server runs the limit, sample and bucket queries it sends. Skipped without pyspark-client, plotly
and pandas.
"""

from __future__ import annotations

from typing import Any

import pytest

import krishiv

pytest.importorskip("pyspark.sql.connect.session", reason="pyspark-client is not installed")
pytest.importorskip("plotly", reason="plotly is not installed")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("grpc")


@pytest.fixture
def spark() -> Any:
    from pyspark.sql import SparkSession

    with krishiv.start_local() as server:
        session = SparkSession.builder.remote(server.url).create()
        try:
            yield session
        finally:
            session.stop()


def _frame(spark: Any) -> Any:
    return spark.createDataFrame([(i, float(i * i)) for i in range(50)], "x int, y double")


def test_max_rows_is_a_server_setting(spark: Any) -> None:
    assert spark.conf.get("spark.sql.pyspark.plotting.max_rows") == "1000"
    spark.conf.set("spark.sql.pyspark.plotting.max_rows", "10")
    assert spark.conf.get("spark.sql.pyspark.plotting.max_rows") == "10"


@pytest.mark.parametrize("kind", ["line", "bar", "barh", "scatter", "area", "pie"])
def test_top_n_plots_collect_the_rows(spark: Any, kind: str) -> None:
    fig = getattr(_frame(spark).plot, kind)(x="x", y="y")
    assert len(fig.data[0].x if kind != "pie" else fig.data[0].values) == 50


def test_top_n_plot_stops_at_max_rows(spark: Any) -> None:
    spark.conf.set("spark.sql.pyspark.plotting.max_rows", "7")
    fig = _frame(spark).plot.bar(x="x", y="y")
    assert len(fig.data[0].x) == 7


def test_kde_plot_is_the_gaussian_density_at_the_given_points(spark: Any) -> None:
    import math

    points, bw = [0.0, 600.0, 2401.0], 100.0
    fig = _frame(spark).plot.kde(column="y", bw_method=bw, ind=points)
    # The oracle: the mean over the 50 values y = x * x of the normal density at each point.
    values = [float(i * i) for i in range(50)]
    want = [
        sum(math.exp(-0.5 * ((p - v) / bw) ** 2) / (bw * math.sqrt(2 * math.pi)) for v in values) / len(values)
        for p in points
    ]
    assert list(fig.data[0].x) == points
    assert list(fig.data[0].y) == pytest.approx(want, rel=1e-9)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "hist needs posexplode in a Connect select list (not a generator there yet),"
        " a two-name alias on it, and array_binary_search"
    ),
)
def test_hist_plot(spark: Any) -> None:
    fig = _frame(spark).plot.hist(column="y", bins=5)
    assert sum(fig.data[0].y) == 50


@pytest.mark.xfail(
    strict=True,
    reason="box sends a WithRelations plan (Spark Connect Relation 41), which Krishiv refuses (T012 group 4)",
)
def test_box_plot(spark: Any) -> None:
    assert _frame(spark).plot.box(column="y").data
