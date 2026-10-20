# SPDX-License-Identifier: Apache-2.0
"""``DataFrame.explain``: PySpark's modes and argument checks, and Krishiv's ``analyze`` and
``verbose`` extensions (``EXPLAIN ANALYZE`` / ``EXPLAIN VERBOSE``) with their JSON documents."""

from __future__ import annotations

import json
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, SparkSession
from krishiv.sql import functions as F

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def df() -> DataFrame:
    spark = SparkSession.builder.create()
    data = spark.createDataFrame(pa.table({"k": pa.array([1, 2, 1, 3], pa.int64()), "v": [1.0, 2.0, 3.0, 4.0]}))
    return data.filter(F.col("v") > 1.5).groupBy("k").agg(F.sum("v").alias("s"))


def test_spark_modes(df: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    df.explain()
    assert capsys.readouterr().out.startswith("== Physical Plan ==\n")
    assert df.explainString(True).startswith("== Parsed Logical Plan ==\n")
    assert df.explainString(mode="formatted").startswith("== Physical Plan ==\n")
    assert df.explainString("cost").startswith("== Optimized Logical Plan ==\n")
    assert df.explainString(mode="CODEGEN") == "Found 0 WholeStageCodegen subtrees.\n"


def test_analyze(df: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    df.explain(analyze=True)
    text = capsys.readouterr().out
    assert text.startswith("== Physical Plan (analyzed) ==\n")
    assert "HashAggregate" in text
    assert "groups=3" in text
    doc = json.loads(df.explainString(analyze=True, json=True))
    assert doc["format"] == "krishiv.query-profile"
    assert doc["version"] == 1
    assert doc["detailed"] is True
    result = [o for o in doc["operators"] if o["node"] is None]
    assert [o["metrics"]["rows_in"] for o in result] == [3]


def test_verbose(df: DataFrame) -> None:
    # A filter over a projection: the optimizer pushes it down.
    df = df.select((F.col("s") * 2).alias("t"), "k").filter(F.col("k") > 1)
    text = df.explainString(verbose=True)
    assert text.startswith("== Parsed Logical Plan ==\n")
    assert "== Optimizer: " in text
    doc = json.loads(df.explainString(verbose=True, json=True))
    assert doc["format"] == "krishiv.explain-verbose"
    assert doc["steps"]
    assert doc["steps"][-1]["plan"] == doc["optimized"]


@pytest.mark.parametrize(
    ("kwargs", "error", "fragment"),
    [
        ({"extended": True, "mode": "simple"}, errors.PySparkValueError, "CANNOT_SET_TOGETHER"),
        ({"analyze": True, "verbose": True}, errors.PySparkValueError, "analyze and verbose"),
        ({"analyze": True, "mode": "simple"}, errors.PySparkValueError, "CANNOT_SET_TOGETHER"),
        ({"json": True}, errors.PySparkValueError, "json and a Spark explain mode"),
        ({"extended": 1}, errors.PySparkTypeError, "NOT_EXPECTED_TYPE"),
        ({"mode": 1}, errors.PySparkTypeError, "NOT_EXPECTED_TYPE"),
        ({"analyze": "yes"}, errors.PySparkTypeError, "NOT_EXPECTED_TYPE"),
    ],
)
def test_bad_arguments(df: DataFrame, kwargs: dict[str, Any], error: type[Exception], fragment: str) -> None:
    with pytest.raises(error, match=fragment):
        df.explainString(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "mode"),
    [({"mode": "nope"}, "nope"), ({"mode": "analyze"}, "analyze"), ({"extended": "VERBOSE"}, "verbose")],
)
def test_unknown_mode(df: DataFrame, kwargs: dict[str, Any], mode: str) -> None:
    # Krishiv's modes are keywords only: a mode string never runs the query.
    with pytest.raises(errors.PySparkValueError, match=f"Unknown explain mode: {mode}"):
        df.explainString(**kwargs)
