# SPDX-License-Identifier: Apache-2.0
"""``krishiv.resultCache.*`` and ``krishiv.deterministicOrder`` through the Spark-style API."""

from __future__ import annotations

import pytest

from krishiv.sql import SparkSession


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def explain(spark: SparkSession, query: str) -> str:
    return str(spark.sql("EXPLAIN " + query).collect()[0][0])


def test_the_plan_text_is_unchanged_while_the_cache_is_off(spark: SparkSession) -> None:
    spark.sql("CREATE TABLE rc_off USING parquet AS SELECT 1 AS a")
    assert "Result Cache" not in explain(spark, "SELECT a FROM rc_off")


def test_a_query_over_a_table_misses_then_hits(spark: SparkSession) -> None:
    spark.conf.set("krishiv.resultCache.enabled", "true")
    spark.sql("CREATE TABLE rc_t USING parquet AS SELECT * FROM VALUES (1), (2), (3) AS v(a)")
    query = "SELECT a FROM rc_t WHERE a > 1"
    assert "== Result Cache ==\nmiss" in explain(spark, query)
    assert [r["a"] for r in spark.sql(query).collect()] == [2, 3]
    assert "hit (metadata_age_ms=" in explain(spark, query)
    # The store's counters follow the first line.
    counters = explain(spark, query).split("== Result Cache ==\n")[1].splitlines()[1]
    assert counters.startswith("store: ") and " hits, " in counters and counters.endswith(" invalidated")
    # Writes through the engine drop the entry.
    spark.sql("INSERT INTO rc_t VALUES (9)")
    assert "== Result Cache ==\nmiss" in explain(spark, query)
    assert sorted(r["a"] for r in spark.sql(query).collect()) == [2, 3, 9]
    spark.sql("REFRESH TABLE rc_t")
    assert "== Result Cache ==\nmiss" in explain(spark, query)


def test_what_cannot_be_cached_says_why(spark: SparkSession) -> None:
    spark.conf.set("krishiv.resultCache.enabled", "true")
    spark.sql("CREATE TABLE rc_n USING parquet AS SELECT 1 AS a")
    assert "not cached: it calls a non-deterministic, clock or session function" in explain(
        spark, "SELECT a, current_timestamp() FROM rc_n"
    )
    spark.createDataFrame([(1,)], ["a"]).createOrReplaceTempView("rc_mem")
    assert "not cached: it reads a table that is not in the catalog" in explain(spark, "SELECT a FROM rc_mem")


def test_the_settings_are_ordinary_session_settings(spark: SparkSession) -> None:
    assert spark.conf.get("krishiv.resultCache.enabled") == "false"
    assert spark.conf.get("krishiv.deterministicOrder") == "false"
    spark.conf.set("krishiv.deterministicOrder", "true")
    assert spark.conf.get("krishiv.deterministicOrder") == "true"
    with pytest.raises(Exception, match=r"krishiv\.resultCache\.ttl"):
        spark.conf.set("krishiv.resultCache.ttl", "not a duration")
