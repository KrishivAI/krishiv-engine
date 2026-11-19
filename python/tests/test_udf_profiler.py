# SPDX-License-Identifier: Apache-2.0
"""The Python UDF profiler (P3-T080): spark.sql.pyspark.udf.profiler = perf | memory, spark.profile."""

from __future__ import annotations

import os

import pytest

from krishiv import SparkSession
from krishiv._udf import profiler
from krishiv.sql.functions import udf


@pytest.fixture
def spark():
    s = SparkSession.builder.getOrCreate()
    profiler.clear()
    yield s
    s.conf.unset("spark.sql.pyspark.udf.profiler")
    profiler.clear()


def _run(spark):
    plus_one = udf(lambda x: x + 1, "long")
    return spark.range(5).select(plus_one("id").alias("v")).collect()


def test_off_by_default_collects_nothing(spark, capsys):
    _run(spark)
    spark.profile.show()
    assert capsys.readouterr().out == ""


def test_perf_profile_shows_and_dumps(spark, capsys, tmp_path):
    spark.conf.set("spark.sql.pyspark.udf.profiler", "perf")
    assert [r.v for r in _run(spark)] == [1, 2, 3, 4, 5]
    spark.profile.show(type="perf")
    out = capsys.readouterr().out
    assert "Profile of UDF<id=" in out
    assert "function calls" in out
    spark.profile.dump(str(tmp_path), type="perf")
    assert any(n.endswith("_perf.pstats") for n in os.listdir(tmp_path))
    spark.profile.clear()
    spark.profile.show()
    assert capsys.readouterr().out == ""


def test_memory_profile_records_a_peak(spark, capsys):
    spark.conf.set("spark.sql.pyspark.udf.profiler", "memory")
    _run(spark)
    spark.profile.show(type="memory")
    out = capsys.readouterr().out
    assert "peak memory:" in out
    assert "calls: 1" in out


def test_unknown_type_is_refused(spark):
    with pytest.raises(Exception, match="type"):
        spark.profile.show(type="flame")


def test_unknown_conf_value_is_refused(spark):
    with pytest.raises(Exception, match="profiler"):
        spark.conf.set("spark.sql.pyspark.udf.profiler", "bogus")
