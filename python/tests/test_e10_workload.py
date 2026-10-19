# SPDX-License-Identifier: Apache-2.0
"""The E10 workload (deep ``withColumn`` chains with ``df.columns`` after each step), through the facade.

Run as a script for the per-step latency: ``python tests/test_e10_workload.py [steps]``.
"""

from __future__ import annotations

import statistics
import sys
import time

import pytest

from krishiv.sql import DataFrame, SparkSession
from krishiv.sql import functions as F

pa = pytest.importorskip("pyarrow")


def chain(spark: SparkSession, steps: int) -> tuple[list[str], list[float]]:
    """``steps`` mixed ``withColumn`` steps (a ``filter`` every 20, ``select("*")`` every 40), ``columns`` after each;
    the final columns and the per-step ``columns`` latencies in seconds."""
    df: DataFrame = spark.createDataFrame(pa.table({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]}))
    latencies = []
    for i in range(steps):
        match i % 5:
            case 0:
                value = F.col("a") + i
            case 1:
                value = F.col("b") - 1
            case 2:
                value = F.col("a") * F.col("b")
            case 3:
                value = F.abs(F.col("a") - i)
            case _:
                value = F.col("a").cast("double")
        df = df.withColumn(f"c{i}", value)
        if i % 20 == 19:
            df = df.filter(F.col("a") > 0)
        if i % 40 == 39:
            df = df.select("*")
        started = time.perf_counter()
        columns = df.columns
        latencies.append(time.perf_counter() - started)
        assert len(columns) == 2 + i + 1
    return df.columns, latencies


def test_withcolumn_chain_columns() -> None:
    spark = SparkSession.builder.create()
    columns, latencies = chain(spark, 60)
    assert columns == ["a", "b", *(f"c{i}" for i in range(60))]
    assert len(latencies) == 60
    # Repeated analysis of the same DataFrame is cheap and stable.
    df = spark.createDataFrame(pa.table({"a": [1]})).withColumn("x", F.col("a") + 1)
    assert [df.columns for _ in range(20)] == [["a", "x"]] * 20
    assert df.schema == df.schema


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    _, lat = chain(SparkSession.builder.create(), n)
    ms = sorted(x * 1000 for x in lat)
    print(f"{n} steps: columns p50 {statistics.median(ms):.2f} ms, p99 {ms[int(len(ms) * 0.99) - 1]:.2f} ms")
