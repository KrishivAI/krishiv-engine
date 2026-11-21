# SPDX-License-Identifier: Apache-2.0
"""The Python UDF profiler (P3-T080): ``spark.sql.pyspark.udf.profiler`` and ``spark.profile``.

Follows ``pyspark/sql/profiler.py`` (4.x): ``show``/``dump``/``clear`` by UDF id and ``type``, ``perf`` output
from ``cProfile``. The ``memory`` type is a summary (``tracemalloc``), not Spark's line-by-line report.
"""

from __future__ import annotations

import os
import pstats
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

import krishiv
from krishiv.sql import SparkSession
from krishiv.sql import functions as F


@pytest.fixture
def spark() -> Iterator[SparkSession]:
    # `krishiv.session()` is the one shared session: start each test with nothing collected and the profiler off.
    session = krishiv.session()
    session.profile.clear()
    session.conf.set("spark.sql.pyspark.udf.profiler", "")
    yield session
    session.conf.set("spark.sql.pyspark.udf.profiler", "")
    session.profile.clear()


def _plus_one():
    @F.udf("long")
    def plus_one(x: int) -> int:
        return x + 1

    return plus_one


def _plus_one_calls(*sessions: SparkSession) -> int:
    """The calls of the ``plus_one`` function in the ``perf`` profiles of ``sessions`` (no profiler bookkeeping)."""
    total = 0
    for session in sessions:
        for entry in session._profile_registry.perf.values():
            if entry.perf is None:
                continue
            raw = getattr(pstats.Stats(entry.perf), "stats")  # noqa: B009 (typeshed lacks the attribute)
            total += sum(row[1] for (_, _, name), row in raw.items() if name == "plus_one")
    return total


def test_off_by_default_collects_nothing(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    plus_one = _plus_one()
    assert [r[0] for r in spark.range(3).select(plus_one("id")).collect()] == [1, 2, 3]
    spark.profile.show()
    assert capsys.readouterr().out == ""
    assert plus_one._unwrapped._profile_id is None  # no id, no wrapper: nothing ran for the profiler


def test_perf_profile_shows_the_udf_calls(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    plus_one = _plus_one()
    spark.conf.set("spark.sql.pyspark.udf.profiler", "perf")
    assert [r[0] for r in spark.range(5).select(plus_one("id")).collect()] == [1, 2, 3, 4, 5]
    spark.profile.show(type="perf")
    out = capsys.readouterr().out
    udf_id = plus_one._unwrapped._profile_id
    assert f"Profile of UDF<id={udf_id}>" in out
    assert "ncalls" in out
    assert "plus_one" in out
    # `show` of the other type prints nothing for a UDF that ran only under `perf`.
    spark.profile.show(type="memory")
    assert capsys.readouterr().out == ""


def test_memory_profile_counts_calls(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    plus_one = _plus_one()
    spark.conf.set("spark.sql.pyspark.udf.profiler", "memory")
    spark.range(4).select(plus_one("id")).collect()
    spark.profile.showMemoryProfiles()
    out = capsys.readouterr().out
    assert "Memory of plus_one: 4 call(s)" in out


def test_show_by_id_and_a_bad_type(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    first, second = _plus_one(), _plus_one()
    spark.conf.set("spark.sql.pyspark.udf.profiler", "perf")
    spark.range(2).select(first("id")).collect()
    spark.range(2).select(second("id")).collect()
    only = first._unwrapped._profile_id
    spark.profile.show(only)
    out = capsys.readouterr().out
    assert f"UDF<id={only}>" in out
    assert f"UDF<id={second._unwrapped._profile_id}>" not in out
    with pytest.raises(ValueError, match="type must be"):
        spark.profile.show(type="flame")


def test_dump_and_clear(spark: SparkSession, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plus_one = _plus_one()
    spark.conf.set("spark.sql.pyspark.udf.profiler", "perf")
    spark.range(3).select(plus_one("id")).collect()
    spark.conf.set("spark.sql.pyspark.udf.profiler", "memory")
    spark.range(3).select(plus_one("id")).collect()
    udf_id = plus_one._unwrapped._profile_id
    spark.profile.dump(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == [f"udf_{udf_id}_memory.json", f"udf_{udf_id}_perf.pstats"]
    stats = pstats.Stats(str(tmp_path / f"udf_{udf_id}_perf.pstats"))
    raw = getattr(stats, "stats")  # noqa: B009 (typeshed lacks the attribute)
    assert [row[1] for (_, _, name), row in raw.items() if name == "plus_one"] == [3]
    spark.profile.clear(udf_id, type="perf")
    spark.profile.show(type="perf")
    assert capsys.readouterr().out == ""
    spark.profile.show(type="memory")
    assert "Memory of plus_one: 3 call(s)" in capsys.readouterr().out
    spark.profile.clear()
    spark.profile.show()
    assert capsys.readouterr().out == ""
    # The profiler keeps collecting after a clear.
    spark.range(2).select(plus_one("id")).collect()
    spark.profile.show()
    assert "Memory of plus_one: 2 call(s)" in capsys.readouterr().out


def test_turning_it_off_stops_collecting(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    plus_one = _plus_one()
    spark.conf.set("spark.sql.pyspark.udf.profiler", "memory")
    spark.range(2).select(plus_one("id")).collect()
    spark.conf.set("spark.sql.pyspark.udf.profiler", "")
    spark.range(7).select(plus_one("id")).collect()
    spark.profile.show()
    assert "2 call(s)" in capsys.readouterr().out


#: How far `cProfile`'s call count may be off on free-threaded CPython, in either direction (seen: +2 on Linux
#: x86_64, -1 on macOS arm64, from a bare `cProfile.Profile` with every enable/disable under one lock).
FREE_THREADED_SLACK = 4


def _count_is_plausible(calls: int, expected: int, *, gil: bool) -> bool:
    """Exact on a GIL build; within `FREE_THREADED_SLACK` of `expected` on a free-threaded one."""
    if gil:
        return calls == expected
    return abs(calls - expected) <= FREE_THREADED_SLACK


@pytest.mark.parametrize(
    ("calls", "gil", "plausible"),
    [
        (80, True, True),
        (79, True, False),  # a lost call on a GIL build is the clash this test exists for
        (81, True, False),
        (80, False, True),
        (79, False, True),
        (82, False, True),
        (84, False, True),
        (75, False, False),  # many lost calls are a clash on a free-threaded build too
        (85, False, False),
    ],
)
def test_the_call_count_check_is_exact_with_a_gil_and_tolerant_without(calls: int, gil: bool, plausible: bool) -> None:
    assert _count_is_plausible(calls, 80, gil=gil) is plausible


def test_profiled_calls_from_many_threads_do_not_clash(spark: SparkSession) -> None:
    plus_one = _plus_one()
    spark.conf.set("spark.sql.pyspark.udf.profiler", "perf")
    errors: list[BaseException] = []

    def run() -> None:
        try:
            spark.range(20).select(plus_one("id")).collect()
        except BaseException as e:  # collect any failure for the assert
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()  # absent before Python 3.13: always a GIL build
    assert _count_is_plausible(_plus_one_calls(spark), 4 * 20, gil=gil)


def test_two_sessions_profiling_at_once_do_not_raise(spark: SparkSession) -> None:
    # `cProfile` is one per process: sessions share one lock, so they take turns instead of failing.
    # Known limit: a UDF reads the profiler setting of the builder's active session, not of the session
    # that runs the query, so this checks only that nothing raises (per-session profiles are a follow-up).
    other = spark.newSession()
    plus_one = _plus_one()
    errors: list[BaseException] = []

    # The UDF expression reads the active session's mode when it is built: set it before any thread.
    spark.conf.set("spark.sql.pyspark.udf.profiler", "perf")

    def run(session: SparkSession) -> None:
        try:
            session.range(30).select(plus_one("id")).collect()
        except BaseException as e:  # collect any failure for the assert
            errors.append(e)

    threads = [threading.Thread(target=run, args=(s,)) for s in (spark, other, spark, other)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # Every call is profiled once, in the registry of one of the two sessions (today the active one's).
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    assert _count_is_plausible(_plus_one_calls(spark, other), 4 * 30, gil=gil)
    other.profile.clear()
