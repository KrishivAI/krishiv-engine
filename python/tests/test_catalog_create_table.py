# SPDX-License-Identifier: Apache-2.0
"""``Catalog.createTable``, ``createExternalTable``, ``recoverPartitions`` and ``dropGlobalTempView`` (KN-10)."""

from __future__ import annotations

from pathlib import Path

import pytest

import krishiv
from krishiv import errors
from krishiv.sql import types as T


@pytest.fixture
def spark():
    return krishiv.session()


def test_create_table_over_a_path_reads_it(spark, tmp_path: Path) -> None:
    path = str(tmp_path / "p")
    spark.sql("select 1 as a, 'x' as b").write.parquet(path)
    df = spark.catalog.createTable("created", path, "parquet")
    assert df.collect()[0].asDict() == {"a": 1, "b": "x"}
    assert spark.catalog.tableExists("created")


def test_create_table_twice_fails(spark, tmp_path: Path) -> None:
    path = str(tmp_path / "p")
    spark.sql("select 1 as a").write.parquet(path)
    spark.catalog.createTable("twice", path, "parquet")
    with pytest.raises(errors.AnalysisException):
        spark.catalog.createTable("twice", path, "parquet")


def test_create_table_with_a_schema(spark, tmp_path: Path) -> None:
    path = str(tmp_path / "p")
    spark.sql("select 1 as a, 'x' as b").write.parquet(path)
    schema = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
    df = spark.catalog.createTable("with_schema", path, "parquet", schema)
    assert df.columns == ["a", "b"]


def test_create_external_table_is_a_deprecated_alias(spark, tmp_path: Path) -> None:
    path = str(tmp_path / "p")
    spark.sql("select 1 as a").write.parquet(path)
    with pytest.warns(FutureWarning):
        df = spark.catalog.createExternalTable("ext", path, "parquet")
    assert df.count() == 1


def test_drop_global_temp_view_says_whether_it_existed(spark) -> None:
    spark.range(3).createOrReplaceGlobalTempView("gview")
    assert spark.catalog.dropGlobalTempView("gview") is True
    assert spark.catalog.dropGlobalTempView("gview") is False


def test_recover_partitions_of_a_plain_table_is_refused(spark, tmp_path: Path) -> None:
    path = str(tmp_path / "p")
    spark.sql("select 1 as a").write.parquet(path)
    spark.catalog.createTable("plain", path, "parquet")
    with pytest.raises(errors.AnalysisException, match="NOT_A_PARTITIONED_TABLE"):
        spark.catalog.recoverPartitions("plain")
