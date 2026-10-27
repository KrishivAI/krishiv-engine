# SPDX-License-Identifier: Apache-2.0
"""``df.write``: Parquet/CSV round trips per save mode, writer options reaching
the writer, and the engine errors for partitioning, unknown formats and modes.

``df.writeTo``: table round trips per provider through ``using``, ``DESCRIBE``,
Spark's existence error classes, and the engine errors for ``INSERT``
(``append`` and the overwrites) and partitioned or clustered tables."""

from __future__ import annotations

import inspect
import os
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, SparkSession
from krishiv.sql import functions as F

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


def test_partition_by_replaces_and_is_forwarded_by_the_shortcuts(df: DataFrame, tmp_path: Any) -> None:
    # Each call replaces the earlier columns, so an empty call clears them.
    path = str(tmp_path / "t")
    df.write.partitionBy("name").partitionBy([]).save(path)
    assert len(df.sparkSession.read.parquet(path).collect()) == 3
    for shortcut in ("parquet", "csv", "json"):
        with pytest.raises(errors.UnsupportedOperationException) as bad:
            getattr(df.write, shortcut)(str(tmp_path / shortcut), partitionBy="name")
        assert "writing partitioned output" in str(bad.value)
    with pytest.raises(errors.UnsupportedOperationException):
        df.write.save(str(tmp_path / "s"), partitionBy=["id"])


def test_none_options_are_skipped(df: DataFrame, tmp_path: Any) -> None:
    path = str(tmp_path / "t")
    df.write.csv(path, header=True, nullValue=None)
    assert len(df.sparkSession.read.csv(path, header=True).collect()) == 3


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
    path = str(tmp_path / "j")
    df.write.json(path)
    assert os.path.exists(os.path.join(path, "_SUCCESS"))

    with pytest.raises(errors.SparkRuntimeException) as bad:
        df.write.format("text").save(str(tmp_path / "text"))
    assert bad.value.getCondition() == "_LEGACY_ERROR_TEMP_1290"

    path = str(tmp_path / "orc")
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.write.format("orc").save(path)
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


def test_writeto_create_parquet_round_trip(df: DataFrame) -> None:
    df.writeTo("t").create()
    back = df.sparkSession.table("t")
    assert [(f.name, f.dataType.simpleString(), f.nullable) for f in back.schema.fields] == [
        ("id", "bigint", True),
        ("name", "string", True),
    ]
    assert dicts(back) == [
        {"id": 1, "name": None},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"},
    ]


def test_writeto_create_csv_round_trip_through_using(df: DataFrame) -> None:
    df.writeTo("t").using("csv").option("header", True).create()
    assert dicts(df.sparkSession.table("t")) == [
        {"id": 1, "name": None},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"},
    ]


def test_writeto_create_json_round_trip_through_using(df: DataFrame) -> None:
    df.writeTo("t").using("json").create()
    assert dicts(df.sparkSession.table("t")) == [
        {"id": 1, "name": None},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"},
    ]


def test_writeto_create_text_round_trip_through_using(spark: SparkSession) -> None:
    lines = spark.createDataFrame(pa.table({"value": ["b", None, "a"]}))
    lines.writeTo("t").using("text").create()
    back = spark.table("t")
    assert [(f.name, f.dataType.simpleString()) for f in back.schema.fields] == [("value", "string")]
    rows = sorted((r.asDict()["value"] for r in back.collect()), key=lambda v: (v is None, v))
    # Text carries no nulls: a NULL writes as an empty line and reads back as
    # an empty string, as in Spark's text source.
    assert rows == ["", "a", "b"]


def test_writeto_describe_shows_the_schema(df: DataFrame) -> None:
    df.writeTo("t").create()
    rows = [r.asDict() for r in df.sparkSession.sql("DESCRIBE TABLE t").collect()]
    assert rows == [
        {"col_name": "id", "data_type": "bigint", "comment": ""},
        {"col_name": "name", "data_type": "string", "comment": ""},
    ]
    with pytest.raises(errors.AnalysisException) as bad:
        df.sparkSession.sql("DESCRIBE TABLE missing").collect()
    assert bad.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"


def test_writeto_create_replace_create_or_replace(df: DataFrame) -> None:
    df.writeTo("t").create()
    with pytest.raises(errors.AnalysisException) as bad:
        df.writeTo("t").create()
    assert bad.value.getCondition() == "TABLE_OR_VIEW_ALREADY_EXISTS"
    assert len(df.sparkSession.table("t").collect()) == 3

    with pytest.raises(errors.AnalysisException) as bad:
        df.writeTo("missing").replace()
    assert bad.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"

    one = df.filter("id = 1")
    one.writeTo("t").replace()
    assert dicts(df.sparkSession.table("t")) == [{"id": 1, "name": None}]

    df.writeTo("u").createOrReplace()
    assert len(df.sparkSession.table("u").collect()) == 3
    one.writeTo("u").createOrReplace()
    assert dicts(df.sparkSession.table("u")) == [{"id": 1, "name": None}]


def test_writeto_failing_create_leaves_no_table(df: DataFrame) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        df.writeTo("t").using("delta").create()
    assert bad.value.getCondition() == "DATA_SOURCE_NOT_FOUND"
    with pytest.raises(errors.AnalysisException) as bad:
        df.sparkSession.table("t").collect()
    assert bad.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"

    with pytest.raises(errors.SparkRuntimeException):
        df.writeTo("u").using("text").create()
    with pytest.raises(errors.AnalysisException) as bad:
        df.sparkSession.table("u").collect()
    assert bad.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"


def test_writeto_append_reports_insert_as_unsupported(df: DataFrame) -> None:
    # Written so it fails visibly when INSERT lands: then no error is raised.
    df.writeTo("t").create()
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.writeTo("t").append()
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "INSERT" in str(bad.value)
    assert len(df.sparkSession.table("t").collect()) == 3


def test_writeto_overwrite_reports_insert_as_unsupported(df: DataFrame) -> None:
    df.writeTo("t").create()
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.writeTo("t").overwrite("id > 1")
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "INSERT" in str(bad.value)
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.writeTo("t").overwrite(F.col("id") > 1)
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert len(df.sparkSession.table("t").collect()) == 3


def test_writeto_overwrite_partitions_reports_insert_as_unsupported(df: DataFrame) -> None:
    df.writeTo("t").create()
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.writeTo("t").overwritePartitions()
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "INSERT" in str(bad.value)
    assert len(df.sparkSession.table("t").collect()) == 3


def test_writeto_partitioned_by_reports_unsupported(df: DataFrame) -> None:
    # Written so it fails visibly when partitioned tables land: then the write succeeds.
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.writeTo("t").partitionedBy("name").create()
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "partitioned tables" in str(bad.value)
    with pytest.raises(errors.AnalysisException) as bad:
        df.sparkSession.table("t").collect()
    assert bad.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"

    df.writeTo("u").partitionedBy("name").partitionedBy([]).create()
    assert len(df.sparkSession.table("u").collect()) == 3


def test_writeto_cluster_by_reports_unsupported(df: DataFrame) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        df.writeTo("t").clusterBy("id").create()
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"
    assert "partitioned tables" in str(bad.value)
    with pytest.raises(errors.AnalysisException) as bad:
        df.sparkSession.table("t").collect()
    assert bad.value.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"


def test_writeto_options_and_table_property(df: DataFrame) -> None:
    df.writeTo("t").using("csv").option("header", True).options(sep="|").tableProperty("k", "v").create()
    assert dicts(df.sparkSession.table("t")) == [
        {"id": 1, "name": None},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"},
    ]


def test_writeto_wrong_types(df: DataFrame) -> None:
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo("t").using(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo("t").option(1, "v")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo("t").tableProperty("k", 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo("t").tableProperty(1, "v")  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo("t").partitionedBy(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo("t").partitionedBy(["id", 1])  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo("t").clusterBy(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.writeTo("t").overwrite(1)  # ty: ignore[invalid-argument-type]


def test_writeto_signatures_match_pyspark() -> None:
    # The PySpark 4.2 parameter names of ``DataFrame.writeTo`` and
    # ``DataFrameWriterV2`` (unverified: from the documented semantics, not a
    # run against PySpark).
    from krishiv.sql import DataFrame as DF
    from krishiv.sql.readwriter import DataFrameWriterV2 as W2

    assert list(inspect.signature(DF.writeTo).parameters) == ["self", "table"]
    assert list(inspect.signature(W2.using).parameters) == ["self", "provider"]
    assert list(inspect.signature(W2.option).parameters) == ["self", "key", "value"]
    assert list(inspect.signature(W2.tableProperty).parameters) == ["self", "property", "value"]
    assert list(inspect.signature(W2.partitionedBy).parameters) == ["self", "cols"]
    assert list(inspect.signature(W2.clusterBy).parameters) == ["self", "cols"]
    assert list(inspect.signature(W2.overwrite).parameters) == ["self", "condition"]
    for name in ("create", "replace", "createOrReplace", "append", "overwritePartitions"):
        assert list(inspect.signature(getattr(W2, name)).parameters) == ["self"], name
