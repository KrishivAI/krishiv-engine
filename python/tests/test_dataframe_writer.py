# SPDX-License-Identifier: Apache-2.0
"""``df.write``: Parquet/CSV round trips per save mode, writer options reaching
the writer, and the engine errors for partitioning, unknown formats and modes."""

from __future__ import annotations

import os
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, SparkSession

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(pa.table({"id": [3, 1, 2], "name": ["c", None, "b"]}))


def dicts(df: DataFrame) -> list[dict[str, Any]]:
    rows = [r.asDict() for r in df.collect()]
    rows.sort(key=lambda r: r["id"])
    return rows


def test_parquet_round_trip_and_success_marker(df: DataFrame, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    df.write.parquet(path)
    assert os.path.exists(os.path.join(path, "_SUCCESS"))
    assert dicts(df.sparkSession.read.parquet(path)) == [
        {"id": 1, "name": None},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"},
    ]


def test_parquet_modes(df: DataFrame, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    df.write.save(path, mode="append")
    df.write.save(path, mode="append")
    assert len(df.sparkSession.read.parquet(path).collect()) == 6

    one = df.filter("id = 1")
    one.write.save(path, mode="overwrite")
    assert dicts(df.sparkSession.read.parquet(path)) == [{"id": 1, "name": None}]

    for mode in ("error", "errorifexists", "default"):
        with pytest.raises(errors.AnalysisException) as bad:
            df.write.save(path, mode=mode)
        assert bad.value.getCondition() == "PATH_ALREADY_EXISTS"
    assert dicts(df.sparkSession.read.parquet(path)) == [{"id": 1, "name": None}]

    df.write.save(path, mode="ignore")
    assert dicts(df.sparkSession.read.parquet(path)) == [{"id": 1, "name": None}]

    fresh = str(tmp_path / "fresh")
    df.write.save(fresh, mode="ignore")
    assert len(df.sparkSession.read.parquet(fresh).collect()) == 3


def test_csv_round_trip_with_options(df: DataFrame, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    df.write.option("header", True).option("sep", "|").csv(path)
    assert os.path.exists(os.path.join(path, "_SUCCESS"))
    back = df.sparkSession.read.option("header", True).option("sep", "|").csv(path)
    # CSV carries no schema: every column reads back as text.
    assert dicts(back) == [
        {"id": "1", "name": None},
        {"id": "2", "name": "b"},
        {"id": "3", "name": "c"},
    ]


def test_save_kwargs_format_mode_options(df: DataFrame, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    df.write.options(header=True).save(path, format="csv", mode="overwrite", sep=";")
    back = df.sparkSession.read.csv(path, header=True, sep=";")
    assert dicts(back) == [
        {"id": "1", "name": None},
        {"id": "2", "name": "b"},
        {"id": "3", "name": "c"},
    ]


def test_save_without_path_raises(df: DataFrame) -> None:
    with pytest.raises(errors.PySparkTypeError) as bad:
        df.write.save()
    assert bad.value.getCondition() == "NOT_EXPECTED_TYPE"


def test_partition_by_raises_unsupported(df: DataFrame, tmp_path: Any) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.write.partitionBy("name").parquet(str(tmp_path / "t"))
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "writing partitioned output" in str(bad.value)

    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.write.partitionBy(["id", "name"]).save(str(tmp_path / "u"))
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"

    with pytest.raises(errors.PySparkTypeError):
        df.write.partitionBy(1)  # ty: ignore[invalid-argument-type]


def test_unknown_format_raises_unsupported(df: DataFrame, tmp_path: Any) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.write.json(str(tmp_path / "j"))
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "writing the json format" in str(bad.value)

    for format in ("text", "orc"):
        with pytest.raises(errors.UnsupportedOperationException) as bad:
            df.write.format(format).save(str(tmp_path / format))
        assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"


def test_unknown_mode_raises_invalid_save_mode(df: DataFrame, tmp_path: Any) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        df.write.mode("bogus").parquet(str(tmp_path / "t"))
    assert bad.value.getCondition() == "INVALID_SAVE_MODE"
