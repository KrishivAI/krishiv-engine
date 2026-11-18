# SPDX-License-Identifier: Apache-2.0
"""Global temporary views (PY-049) and ``interruptAll`` / ``interruptTag`` / ``interruptOperation`` /
``addTag`` (PY-056)."""

from __future__ import annotations

import threading
import time

import pytest

import krishiv
from krishiv.errors import AnalysisException, PySparkValueError


@pytest.fixture
def spark() -> krishiv.SparkSession:
    return krishiv.SparkSession.builder.create()


def test_a_global_temp_view_is_read_as_global_temp_name(spark: krishiv.SparkSession) -> None:
    spark.range(3).createGlobalTempView("gv")
    assert spark.sql("SELECT count(*) AS n FROM global_temp.gv").collect()[0]["n"] == 3
    assert spark.table("global_temp.gv").count() == 3


def test_a_global_temp_view_is_not_a_local_one(spark: krishiv.SparkSession) -> None:
    spark.range(3).createGlobalTempView("gv2")
    with pytest.raises(AnalysisException):
        spark.sql("SELECT * FROM gv2").collect()


def test_creating_an_existing_global_view_fails_and_replace_replaces(spark: krishiv.SparkSession) -> None:
    spark.range(3).createGlobalTempView("gv3")
    with pytest.raises(AnalysisException) as info:
        spark.range(5).createGlobalTempView("gv3")
    assert info.value.getCondition() == "TEMP_TABLE_OR_VIEW_ALREADY_EXISTS"
    assert "`global_temp`.`gv3`" in str(info.value)
    spark.range(5).createOrReplaceGlobalTempView("gv3")
    assert spark.table("global_temp.gv3").count() == 5


def test_a_global_view_belongs_to_its_session(spark: krishiv.SparkSession) -> None:
    # Spark shares `global_temp` across an application's sessions; Krishiv keeps it per session
    # (docs/concepts/known-differences.md, "Tags and global views").
    spark.range(2).createGlobalTempView("gv4")
    with pytest.raises(AnalysisException):
        spark.newSession().table("global_temp.gv4").count()


def test_a_bad_query_is_reported_before_an_existing_name(spark: krishiv.SparkSession) -> None:
    spark.range(1).createGlobalTempView("gv5")
    bad = spark.sql("SELECT 1 AS x").selectExpr("nope")
    with pytest.raises(AnalysisException) as info:
        bad.createGlobalTempView("gv5")
    assert info.value.getCondition() != "TEMP_TABLE_OR_VIEW_ALREADY_EXISTS"


def test_replacing_a_global_view_drops_its_cached_data(spark: krishiv.SparkSession) -> None:
    spark.sql("SELECT 1 AS x").createGlobalTempView("gv6")
    spark.sql("CACHE TABLE global_temp.gv6")
    assert spark.catalog.isCached("global_temp.gv6")
    spark.sql("SELECT 2 AS x").createOrReplaceGlobalTempView("gv6")
    assert not spark.catalog.isCached("global_temp.gv6")


def test_interrupt_with_nothing_running_returns_nothing(spark: krishiv.SparkSession) -> None:
    assert spark.interruptAll() == []
    assert spark.interruptTag("nobody") == []
    assert spark.interruptOperation("00000000-0000-0000-0000-000000000000") == []


def test_interrupt_of_an_id_or_tag_nothing_has_cancels_nothing(spark: krishiv.SparkSession) -> None:
    assert spark.interruptOperation("not-an-operation-id") == []
    assert spark.interruptTag("") == []
    assert spark.interruptTag("a,b") == []


def test_tags_are_per_thread_and_validated(spark: krishiv.SparkSession) -> None:
    spark.addTag("a")
    spark.addTag("b")
    assert spark.getTags() == {"a", "b"}
    seen: list[set[str]] = []
    worker = threading.Thread(target=lambda: seen.append(set(spark.getTags())))
    worker.start()
    worker.join()
    assert seen == [set()]
    spark.removeTag("a")
    spark.removeTag("absent")
    assert spark.getTags() == {"b"}
    spark.clearTags()
    assert spark.getTags() == set()
    for bad in ("", "a,b"):
        with pytest.raises(PySparkValueError):
            spark.addTag(bad)
    with pytest.raises(PySparkValueError):
        spark.addTag(None)  # ty: ignore[invalid-argument-type]


def test_tagged_frames_still_run(spark: krishiv.SparkSession) -> None:
    spark.addTag("work")
    assert spark.range(4).count() == 4
    assert spark.sql("SELECT 1 AS x").collect()[0]["x"] == 1
    spark.clearTags()


def test_interrupt_tag_cancels_a_long_query_and_spares_others(spark: krishiv.SparkSession) -> None:
    outcome: list[object] = []
    started = threading.Event()

    def run() -> None:
        spark.addTag("long")
        df = spark.sql("SELECT sum(id) AS s FROM range(0, 3000000000)")
        started.set()
        try:
            outcome.append(df.collect())
        except Exception as error:  # the cancellation
            outcome.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    started.wait(5)
    # Let the query reach execution: a cancel during analysis ends it too, but lists no id.
    time.sleep(1.0)
    cancelled: list[str] = []
    deadline = time.monotonic() + 20
    while not cancelled and time.monotonic() < deadline:
        cancelled = spark.interruptTag("long")
        time.sleep(0.05)
    worker.join(30)
    assert not worker.is_alive()
    assert len(cancelled) == 1
    assert isinstance(outcome[0], Exception)
    assert "OPERATION_CANCELED" in str(outcome[0])
    assert spark.range(3).count() == 3
