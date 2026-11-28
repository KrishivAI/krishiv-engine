# SPDX-License-Identifier: Apache-2.0
"""``F.window`` / ``F.window_time`` (QA2-26): the Column API over the engine's time windows."""

from __future__ import annotations

import datetime as dt

import pytest

from krishiv.sql import SparkSession
from krishiv.sql import functions as F


@pytest.fixture(scope="module")
def df():
    spark = SparkSession.builder.getOrCreate()
    return spark.sql("SELECT timestamp'2024-01-01 00:00:03' AS ts, 1 AS v")


def test_tumbling_window_groups_by_bucket(df) -> None:
    (row,) = df.groupBy(F.window("ts", "5 seconds")).count().collect()
    assert row["window"]["start"] == dt.datetime(2024, 1, 1, 0, 0, 0)
    assert row["window"]["end"] == dt.datetime(2024, 1, 1, 0, 0, 5)
    assert row["count"] == 1


def test_sliding_window_puts_a_row_in_each_overlapping_window(df) -> None:
    rows = df.select(F.window_time(F.window(df.ts, "4 seconds", "2 seconds", "1 second")).alias("t")).collect()
    assert sorted(r["t"] for r in rows) == [
        dt.datetime(2024, 1, 1, 0, 0, 4, 999999),
        dt.datetime(2024, 1, 1, 0, 0, 6, 999999),
    ]


def test_start_time_alone_tumbles(df) -> None:
    (row,) = df.select(F.window("ts", "5 seconds", startTime="1 second").alias("w")).collect()
    assert row["w"]["start"] == dt.datetime(2024, 1, 1, 0, 0, 1)


def test_durations_must_be_strings(df) -> None:
    with pytest.raises(TypeError):
        F.window("ts", 5)  # ty: ignore[invalid-argument-type]
