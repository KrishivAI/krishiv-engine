# SPDX-License-Identifier: Apache-2.0
"""Delta tables are built into the wheel: the CI wheel build once had no ``delta`` feature, so every Delta write,
DML and MERGE failed with DATA_SOURCE_NOT_FOUND. This writes and reads a table through the installed wheel."""

from __future__ import annotations

from pathlib import Path

import pytest

from krishiv.sql import SparkSession


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def test_a_delta_table_is_written_appended_and_read_back(spark: SparkSession, tmp_path: Path) -> None:
    path = str(tmp_path / "t")
    spark.createDataFrame([(1, "a"), (2, "b")], ["id", "v"]).write.format("delta").save(path)
    spark.createDataFrame([(3, "c")], ["id", "v"]).write.format("delta").mode("append").save(path)
    rows = spark.read.format("delta").load(path).orderBy("id").collect()
    assert [(r["id"], r["v"]) for r in rows] == [(1, "a"), (2, "b"), (3, "c")]
    assert (tmp_path / "t" / "_delta_log").is_dir()


def test_delta_dml_and_merge_run_in_the_wheel(spark: SparkSession, tmp_path: Path) -> None:
    path = str(tmp_path / "d")
    spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "v"]).write.format("delta").save(path)
    spark.sql(f"UPDATE delta.`{path}` SET v = 'x' WHERE id = 1")
    spark.sql(f"DELETE FROM delta.`{path}` WHERE id = 2")
    spark.createDataFrame([(3, "m"), (4, "n")], ["id", "v"]).createOrReplaceTempView("src")
    spark.sql(
        f"MERGE INTO delta.`{path}` t USING src s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET v = s.v WHEN NOT MATCHED THEN INSERT *"
    )
    rows = spark.read.format("delta").load(path).orderBy("id").collect()
    assert [(r["id"], r["v"]) for r in rows] == [(1, "x"), (3, "m"), (4, "n")]
