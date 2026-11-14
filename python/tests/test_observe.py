# SPDX-License-Identifier: Apache-2.0
"""``DataFrame.observe`` and ``Observation`` (P3-T084), against the Spark 4.2 behaviour."""

from __future__ import annotations

import threading

import pytest

from krishiv.errors import PySparkAssertionError, PySparkTypeError, PySparkValueError
from krishiv.sql import DataFrame, Observation, SparkSession
from krishiv.sql import functions as F

pytest.importorskip("pyarrow")


@pytest.fixture
def df() -> DataFrame:
    return SparkSession.builder.create().range(0, 10)


def test_basic(df: DataFrame) -> None:
    obs = Observation("stats")
    out = df.observe(obs, F.count(F.lit(1)).alias("n"), F.max("id").alias("m"))
    assert out.count() == 10
    assert obs.get == {"n": 10, "m": 9}


def test_unnamed_and_unaliased_names(df: DataFrame) -> None:
    obs = Observation()
    df.observe(obs, F.count(F.lit(1)), F.max("id")).collect()
    assert obs.get == {"count(1)": 10, "max(id)": 9}


def test_first_action_wins(df: DataFrame) -> None:
    obs = Observation()
    out = df.observe(obs, F.count(F.lit(1)).alias("n"))
    out.count()
    out.filter(F.col("id") > 5).count()
    assert obs.get == {"n": 10}


def test_get_blocks_until_action(df: DataFrame) -> None:
    obs = Observation()
    out = df.observe(obs, F.count(F.lit(1)).alias("n"))
    seen = []
    t = threading.Thread(target=lambda: seen.append(obs.get))
    t.start()
    t.join(0.2)
    assert t.is_alive()
    out.count()
    t.join(10)
    assert seen == [{"n": 10}]


def test_string_name_form(df: DataFrame) -> None:
    assert df.observe("m", F.count(F.lit(1)).alias("n")).count() == 10


def test_reuse(df: DataFrame) -> None:
    obs = Observation()
    df.observe(obs, F.count(F.lit(1)))
    with pytest.raises(PySparkAssertionError) as e:
        df.observe(obs, F.count(F.lit(1)))
    assert e.value.getCondition() == "REUSE_OBSERVATION"


def test_get_before_observe() -> None:
    with pytest.raises(PySparkAssertionError) as e:
        _ = Observation().get
    assert e.value.getCondition() == "NO_OBSERVE_BEFORE_GET"


def test_argument_errors(df: DataFrame) -> None:
    with pytest.raises(PySparkValueError) as e:
        Observation("")
    assert e.value.getCondition() == "VALUE_NOT_NON_EMPTY_STR"
    with pytest.raises(PySparkTypeError) as t:
        Observation(5)  # ty: ignore[invalid-argument-type]
    assert t.value.getCondition() == "NOT_EXPECTED_TYPE"
    with pytest.raises(PySparkValueError) as c:
        df.observe(Observation())
    assert c.value.getCondition() == "CANNOT_BE_EMPTY"
    with pytest.raises(PySparkValueError) as c2:
        df.observe("m")
    assert c2.value.getCondition() == "CANNOT_BE_EMPTY"


def test_limit_zero_gives_empty(df: DataFrame) -> None:
    obs = Observation()
    df.observe(obs, F.count(F.lit(1)).alias("n")).limit(0).collect()
    assert obs.get == {}
