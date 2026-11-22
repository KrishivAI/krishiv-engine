# SPDX-License-Identifier: Apache-2.0
"""Fixes from the review of the R4 acceptance PR (#973)."""

from __future__ import annotations

import array
import datetime

import pytest

from krishiv.sql import Row, SparkSession
from krishiv.sql import types as T


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def test_tree_string_takes_max_depth_keyword_and_escapes_labels() -> None:
    nested = T.StructType([T.StructField("a\nb", T.StructType([T.StructField("x", T.IntegerType())]))])
    assert nested.treeString(maxDepth=1) == "root\n |-- a\\nb: struct (nullable = true)\n"


def test_unsigned_array_typecodes_infer(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(array.array("B", [255]),), (array.array("H", [7]),)][:1])
    assert frame.schema.fields[0].dataType == T.ArrayType(T.ShortType(), False)


def test_zero_column_arrow_table_keeps_its_rows(spark: SparkSession) -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": [1, 2, 3]}).select([])
    assert spark.createDataFrame(table).count() == 3


def test_nanosecond_timestamps_truncate_to_microseconds(spark: SparkSession) -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"t": pa.array([1001, 2000], type=pa.timestamp("ns", tz="UTC"))})
    got = [r[0] for r in spark.createDataFrame(table).collect()]
    assert [t.timestamp() for t in got] == [0.000001, 0.000002]


def test_explicit_ntz_schema_keeps_wall_clock(spark: SparkSession) -> None:
    pa = pytest.importorskip("pyarrow")
    spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    try:
        table = pa.table({"t": pa.array([datetime.datetime(2024, 1, 15, 12, 0)], type=pa.timestamp("us"))})
        got = spark.createDataFrame(table, "t timestamp_ntz").collect()[0][0]
        assert got == datetime.datetime(2024, 1, 15, 12, 0)
    finally:
        spark.conf.unset("spark.sql.session.timeZone")


def test_approx_quantile_of_no_columns_is_empty(spark: SparkSession) -> None:
    assert spark.range(3).stat.approxQuantile([], [0.5], 0.0) == []


def test_sampling_survives_a_column_named_like_the_temporary(spark: SparkSession) -> None:
    frame = spark.range(5).withColumnRenamed("id", "__krishiv_sample_rand")
    out = frame.sample(fraction=1.0, seed=1)
    assert out.columns == ["__krishiv_sample_rand"]
    assert out.count() == 5


def test_wholetext_false_resets_an_earlier_true(spark: SparkSession, tmp_path) -> None:
    (tmp_path / "t.txt").write_text("a\nb\n")
    reader = spark.read
    assert reader.text(str(tmp_path), wholetext=True).count() == 1
    assert reader.text(str(tmp_path), wholetext=False).count() == 2


def test_pivot_with_an_explicit_empty_list_does_not_discover(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(1, "a")], "g int, p string")
    spark.conf.set("spark.sql.pivotMaxValues", "0")
    try:
        frame.groupBy("g").pivot("p", []).count()
    finally:
        spark.conf.unset("spark.sql.pivotMaxValues")


def test_to_turns_a_null_typed_column_into_a_typed_null(spark: SparkSession) -> None:
    out = spark.sql("SELECT NULL AS x").to(T.StructType([T.StructField("x", T.IntegerType(), True)]))
    assert out.schema["x"].dataType == T.IntegerType()
    assert out.collect() == [Row(x=None)]


def test_summary_with_a_repeated_statistic_under_last_win(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(1,), (2,)], "a int")
    with pytest.raises(Exception, match="DUPLICATED_MAP_KEY"):
        frame.summary("count", "count")
    spark.conf.set("spark.sql.mapKeyDedupPolicy", "LAST_WIN")
    try:
        assert [r[0] for r in frame.summary("count", "count").collect()] == ["count"]
    finally:
        spark.conf.unset("spark.sql.mapKeyDedupPolicy")


def test_random_split_keeps_array_and_struct_columns_in_the_sort_key(spark: SparkSession) -> None:
    frame = spark.createDataFrame(
        [([1], (1,)), ([2], (2,))], "a array<int>, s struct<x:int>"
    )  # the list column is not a sort key
    assert sum(part.count() for part in frame.randomSplit([1.0, 1.0], 7)) == 2


def test_stats_refuse_a_string_column_named_by_a_quoted_or_qualified_reference(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("1", 2), ("3", 4)], "s string, n int").alias("t")
    for ref in ("`s`", "t.s"):
        with pytest.raises(Exception, match="not supported"):
            frame.stat.corr(ref, "n")
        with pytest.raises(Exception, match="not supported"):
            frame.stat.approxQuantile(ref, [0.5], 0.0)


def test_ntz_value_with_an_offset_keeps_its_wall_clock(spark: SparkSession) -> None:
    import datetime as dt

    value = dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    frame = spark.createDataFrame([(value,)], "t timestamp_ntz")
    assert frame.collect()[0][0] == dt.datetime(2024, 1, 15, 12, 0)


def test_pivot_accepts_none_among_explicit_values(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(1, "a", 1), (1, None, 2)], "g int, p string, v int")
    rows = frame.groupBy("g").pivot("p", [None, "a"]).sum("v").collect()
    assert len(rows) == 1
    assert 2 in tuple(rows[0])
    assert 1 in tuple(rows[0])


def test_a_negative_nanosecond_timestamp_floors_to_microseconds(spark: SparkSession) -> None:
    import pyarrow as pa

    table = pa.table({"t": pa.array([-1001, None], type=pa.timestamp("ns", tz="UTC"))})
    values = spark.createDataFrame(table).collect()
    assert values[0][0].timestamp() == pytest.approx(-0.000002)
