# SPDX-License-Identifier: Apache-2.0
"""``df.cache`` / ``persist`` / ``unpersist`` and the catalog calls behind them."""

from __future__ import annotations

import pytest

from krishiv.sql import DataFrame, SparkSession
from krishiv.storagelevel import StorageLevel

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def rows(df: DataFrame) -> list[tuple[int, str | None]]:
    out = [(r["age"], r["name"]) for r in df.collect()]
    out.sort()
    return out


def test_cache_and_unpersist(spark: SparkSession) -> None:
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    assert not df.is_cached
    assert df.storageLevel == StorageLevel(False, False, False, False)
    df = df.cache()
    assert df.storageLevel == StorageLevel.MEMORY_AND_DISK
    assert rows(df) == [(2, "Alice"), (5, "Bob")]
    assert df.is_cached
    assert df.count() == 2
    df = df.unpersist()
    assert not df.is_cached
    assert df.storageLevel == StorageLevel(False, False, False, False)
    assert rows(df) == [(2, "Alice"), (5, "Bob")]


def test_cache_acts_on_the_frame_itself(spark: SparkSession) -> None:
    """PySpark's idiom: ``df.cache()`` on its own line marks ``df`` and returns it."""
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    assert df.cache() is df
    assert df.count() == 2
    assert df.is_cached
    assert df.persist(StorageLevel.DISK_ONLY) is df
    assert df.storageLevel == StorageLevel.DISK_ONLY
    assert df.unpersist() is df
    assert not df.is_cached
    assert df.count() == 2
    assert not df.is_cached


def test_persist_levels(spark: SparkSession) -> None:
    df = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
    for level in (
        StorageLevel.MEMORY_ONLY,
        StorageLevel.MEMORY_ONLY_2,
        StorageLevel.MEMORY_AND_DISK,
        StorageLevel.MEMORY_AND_DISK_2,
        StorageLevel.DISK_ONLY,
        StorageLevel.DISK_ONLY_2,
        StorageLevel.OFF_HEAP,
    ):
        df = df.persist(level)
        assert df.storageLevel == level
        assert df.count() == 2
        assert df.is_cached
        df = df.unpersist()
        assert not df.is_cached
    with pytest.raises(Exception, match="storageLevel"):
        df.persist("MEMORY_ONLY")  # ty: ignore[invalid-argument-type]


def test_cached_frame_reads_the_snapshot(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, "a")], ["age", "name"])
    df.createOrReplaceTempView("t")
    cached = spark.table("t").cache()
    assert cached.count() == 1
    # Replacing the view re-analyzes to a new plan, so the next action reads
    # the new definition (the old snapshot is orphaned, not served).
    spark.createDataFrame([(2, "b")], ["age", "name"]).createOrReplaceTempView("t")
    assert rows(cached) == [(2, "b")]
    spark.catalog.refreshTable("t")
    assert not cached.is_cached
    assert rows(cached) == [(2, "b")]
    assert cached.is_cached
    spark.catalog.clearCache()
    assert not cached.is_cached


def test_catalog_cache_calls(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, "a")], ["age", "name"])
    df.createOrReplaceTempView("t")
    assert not spark.catalog.isCached("t")
    assert not spark.catalog.isCached("missing")
    spark.catalog.cacheTable("t")
    assert spark.catalog.isCached("t")
    spark.catalog.uncacheTable("t")
    assert not spark.catalog.isCached("t")
    spark.catalog.cacheTable("t", StorageLevel.DISK_ONLY)
    assert spark.catalog.isCached("t")
    spark.catalog.clearCache()
    assert not spark.catalog.isCached("t")


def test_sql_clear_cache(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, "a")], ["age", "name"])
    df.createOrReplaceTempView("t")
    cached = spark.table("t").cache()
    assert cached.count() == 1
    spark.sql("CLEAR CACHE")
    assert not cached.is_cached


def test_catalog_cache_calls_check_argument_types(spark: SparkSession) -> None:
    catalog = spark.catalog
    for call in (
        catalog.uncacheTable,
        catalog.refreshTable,
        catalog.refreshByPath,
        catalog.isCached,
        catalog.cacheTable,
    ):
        with pytest.raises(TypeError):
            call(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        catalog.cacheTable("t", "MEMORY_ONLY")  # ty: ignore[invalid-argument-type]


def test_refresh_by_path_and_refresh_table(spark: SparkSession) -> None:
    spark.createDataFrame([(1, "a")], ["age", "name"]).createOrReplaceTempView("t")
    cached = spark.table("t").cache()
    assert cached.count() == 1
    spark.catalog.refreshByPath("/no/such/dir")
    assert spark.catalog.refreshTable("t").count() == 1


def test_storage_level_repr_str_and_equality() -> None:
    level = StorageLevel(True, True, False, True, 2)
    assert repr(level) == "StorageLevel(True, True, False, True, 2)"
    assert str(level) == "Disk Deserialized 2x Replicated"
    assert str(StorageLevel.MEMORY_ONLY) == "Memory Deserialized 1x Replicated"
    assert str(StorageLevel.OFF_HEAP) == "OffHeap Serialized 1x Replicated"
    assert str(StorageLevel(False, False, False, False)) == "Serialized 1x Replicated"
    assert level == StorageLevel(True, True, False, True, 2)
    assert level != StorageLevel.MEMORY_AND_DISK
    assert level != "Disk"
