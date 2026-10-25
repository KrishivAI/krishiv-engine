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


def test_partition_by_empty_is_a_noop(df: DataFrame, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    df.write.partitionBy([]).save(path)
    assert os.path.exists(os.path.join(path, "_SUCCESS"))
    assert len(df.sparkSession.read.parquet(path).collect()) == 3


def test_invalid_frame_reports_analysis_error_before_partitioning(spark: SparkSession, tmp_path: Any) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        spark.table("missing").write.partitionBy("name").save(str(tmp_path / "t"))
    assert bad.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"
    assert not os.path.exists(str(tmp_path / "t"))


def test_partition_by_with_bogus_mode_raises_invalid_save_mode(df: DataFrame, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    with pytest.raises(errors.AnalysisException) as bad:
        df.write.partitionBy("name").mode("bogus").save(path)
    assert bad.value.getCondition() == "INVALID_SAVE_MODE"
    assert not os.path.exists(path)


def test_unknown_format_raises_unsupported(df: DataFrame, tmp_path: Any) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.write.json(str(tmp_path / "j"))
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "writing the json format" in str(bad.value)

    for format in ("text", "orc"):
        path = str(tmp_path / format)
        with pytest.raises(errors.UnsupportedOperationException) as bad:
            df.write.format(format).save(path)
        assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
        assert not os.path.exists(path)


def test_csv_modes(df: DataFrame, tmp_path: Any) -> None:
    def rows(frame: DataFrame) -> list[tuple[str, Any]]:
        out = [(r.asDict()["_c0"], r.asDict()["_c1"]) for r in frame.collect()]
        out.sort(key=lambda r: (r[0] is None, r[0]))
        return out

    path = str(tmp_path / "t")
    df.write.format("csv").save(path, mode="append")
    df.write.format("csv").save(path, mode="append")
    assert len(df.sparkSession.read.csv(path).collect()) == 6

    one = df.filter("id = 1")
    one.write.format("csv").save(path, mode="overwrite")
    assert rows(df.sparkSession.read.csv(path)) == [("1", None)]

    for mode in ("error", "errorifexists", "default"):
        with pytest.raises(errors.AnalysisException) as bad:
            df.write.format("csv").save(path, mode=mode)
        assert bad.value.getCondition() == "PATH_ALREADY_EXISTS"
    assert rows(df.sparkSession.read.csv(path)) == [("1", None)]

    df.write.format("csv").save(path, mode="ignore")
    assert rows(df.sparkSession.read.csv(path)) == [("1", None)]

    fresh = str(tmp_path / "fresh")
    df.write.format("csv").save(fresh, mode="ignore")
    assert len(df.sparkSession.read.csv(fresh).collect()) == 3


def test_csv_compression_option_raises_unsupported(df: DataFrame, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.write.option("compression", "gzip").csv(path)
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "compression codec gzip" in str(bad.value)
    assert not os.path.exists(path)


def test_unknown_mode_raises_invalid_save_mode(df: DataFrame, tmp_path: Any) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        df.write.mode("bogus").parquet(str(tmp_path / "t"))
    assert bad.value.getCondition() == "INVALID_SAVE_MODE"
