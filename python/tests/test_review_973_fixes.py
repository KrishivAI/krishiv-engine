# SPDX-License-Identifier: Apache-2.0
"""Fixes from the review of the R4 acceptance PR (#973)."""

from __future__ import annotations

import array
import datetime
import math

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
    assert [t.microsecond for t in got] == [1, 2]  # no .timestamp(): Windows rejects a pre-1970 local time


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
    (tmp_path / "t.txt").write_text("a\nb\n", newline="\n")
    reader = spark.read
    assert reader.text(str(tmp_path), wholetext=True).count() == 1
    assert reader.text(str(tmp_path), wholetext=False).count() == 2


def test_pivot_with_an_explicit_empty_list_does_not_discover(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(1, "a")], "g int, p string")
    spark.conf.set("spark.sql.pivotMaxValues", "0")
    try:
        assert frame.groupBy("g").pivot("p", []).count().collect() == [Row(g=1)]
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
        assert [r[0] for r in frame.summary("count", "count").collect()] == ["count", "count"]
    finally:
        spark.conf.unset("spark.sql.mapKeyDedupPolicy")


def test_random_split_keeps_array_and_struct_columns_in_the_sort_key(spark: SparkSession) -> None:
    rows = [(k, [k], (k,)) for k in (1, 2, 3, 4)]
    schema = "k int, a array<int>, s struct<x:int>"
    first = [
        [tuple(r) for r in p.collect()]
        for p in spark.createDataFrame(rows, schema).coalesce(1).randomSplit([1.0, 1.0], 7)
    ]
    again = [
        [tuple(r) for r in p.collect()]
        for p in spark.createDataFrame(rows[::-1], schema).coalesce(1).randomSplit([1.0, 1.0], 7)
    ]
    # The array and struct columns ride along while the sort is by `k`: the split does not depend on the input
    # order, so the same seed gives the same rows in each part.
    assert [sorted(p) for p in again] == [sorted(p) for p in first]
    assert sum(len(p) for p in first) == 4


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
    assert [tuple(r) for r in rows] == [(1, 2, 1)]  # g, the sum under NULL, the sum under "a"


def test_a_negative_nanosecond_timestamp_floors_to_microseconds(spark: SparkSession) -> None:
    import pyarrow as pa

    table = pa.table({"t": pa.array([-1001, None], type=pa.timestamp("ns", tz="UTC"))})
    values = spark.createDataFrame(table).collect()
    assert values[0][0].timestamp() == pytest.approx(-0.000002)


def test_an_arrow_map_column_survives_the_python_value_fallback(spark: SparkSession) -> None:
    import pyarrow as pa

    table = pa.table(
        {
            "m": pa.array([[("a", 1)], None], type=pa.map_(pa.string(), pa.int64())),
            "s": pa.array([{"k": 1}, {"k": 2}], type=pa.struct([("k", pa.int64())])),
        }
    )
    out = spark.createDataFrame(table, "m map<string,bigint>, s map<string,bigint>").collect()
    assert [r["m"] for r in out] == [{"a": 1}, None]
    assert [r["s"] for r in out] == [{"k": 1}, {"k": 2}]


def test_a_non_container_for_an_array_is_null_without_verification(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(42,), ([1, 2],)], "a array<int>", verifySchema=False)
    assert [r[0] for r in frame.collect()] == [None, [1, 2]]


def test_a_bad_array_element_is_null_and_the_rest_stay(spark: SparkSession) -> None:
    frame = spark.createDataFrame([([1, "bad", 3],)], "a array<int>", verifySchema=False)
    assert frame.collect()[0][0] == [1, None, 3]


def test_random_split_weights_whose_sum_overflows(spark: SparkSession) -> None:
    frame = spark.range(100)
    parts = frame.randomSplit([1e308, 1e308], 7)
    assert sum(p.count() for p in parts) == 100


def test_pivot_discovery_over_a_qualified_column(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(1, "a", 1), (1, "b", 2)], "g int, p string, v int").alias("s")
    rows = frame.groupBy("g").pivot("s.p").sum("v").collect()
    assert [tuple(r) for r in rows] == [(1, 1, 2)]


def test_to_keeps_a_non_nullable_struct_non_nullable(spark: SparkSession) -> None:
    from krishiv.sql.types import IntegerType, StructField, StructType

    inner = StructType([StructField("x", IntegerType(), False)])
    schema = StructType([StructField("s", inner, False)])
    frame = spark.createDataFrame([((1,),)], schema)
    assert frame.to(schema).to(schema).collect()[0][0][0] == 1


def test_explicit_timestamp_target_localizes_under_ntz_inference(spark: SparkSession) -> None:
    import datetime as dt

    import pyarrow as pa

    old_type = spark.conf.get("spark.sql.timestampType", "TIMESTAMP_LTZ") or "TIMESTAMP_LTZ"
    old_zone = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.timestampType", "TIMESTAMP_NTZ")
    spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    try:
        table = pa.table({"t": pa.array([dt.datetime(2024, 1, 15, 12, 0)])})
        frame = spark.createDataFrame(table, "t timestamp")
        assert frame.collect()[0][0].astimezone(dt.UTC).hour == 20
    finally:
        spark.conf.set("spark.sql.timestampType", old_type)
        if old_zone is None:
            spark.conf.unset("spark.sql.session.timeZone")
        else:
            spark.conf.set("spark.sql.session.timeZone", old_zone)


def test_unchecked_double_target_takes_only_floats(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(1,), (1.5,)], "x double", verifySchema=False)
    assert [r[0] for r in frame.collect()] == [None, 1.5]


def test_unsigned_arrow_column_is_cast_to_an_explicit_target(spark: SparkSession) -> None:
    import pyarrow as pa

    frame = spark.createDataFrame(pa.table({"x": pa.array([1, 2], type=pa.uint32())}), "x bigint")
    assert [r[0] for r in frame.collect()] == [1, 2]


def test_sample_by_matches_a_boolean_stratum_only_with_boolean_keys(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(True,), (False,)], "b boolean")
    assert frame.sampleBy("b", {1: 1.0}, 1).count() == 0
    assert frame.sampleBy("b", {True: 1.0}, 1).count() == 1


def test_unchecked_string_for_array_is_null_not_its_characters(spark: SparkSession) -> None:
    rows = spark.createDataFrame([("ab",), ([1, 2],)], "a array<string>", verifySchema=False).collect()
    assert [r.a for r in rows] == [None, ["1", "2"]]


def test_unchecked_integer_narrows_to_the_target_width(spark: SparkSession) -> None:
    df = spark.createDataFrame([(128,), (-129,), (5,)], "a tinyint", verifySchema=False)
    assert [r.a for r in df.collect()] == [-128, 127, 5]
    df = spark.createDataFrame([(2**31,)], "a int", verifySchema=False)
    assert df.collect()[0].a == -(2**31)


def test_freq_items_counts_equal_arrays_with_nan_as_one_item(spark: SparkSession) -> None:
    nan = float("nan")
    df = spark.createDataFrame([([nan],), ([nan],)], "a array<double>")
    items = df.freqItems(["a"], 1.0).collect()[0][0]
    assert len(items) == 1
    assert len(items[0]) == 1
    assert math.isnan(items[0][0])


def test_unchecked_integer_beyond_the_long_range_is_null(spark: SparkSession) -> None:
    df = spark.createDataFrame([(2**64 + 1,), (2**63,), (5,)], "x bigint", verifySchema=False)
    assert [r.x for r in df.collect()] == [None, None, 5]


def test_an_attribute_object_is_a_struct_value(spark: SparkSession) -> None:
    from types import SimpleNamespace

    df = spark.createDataFrame([(SimpleNamespace(x=1),)], "s struct<x:int>")
    assert df.collect()[0]["s"]["x"] == 1


def test_decimal_nan_is_null(spark: SparkSession) -> None:
    from decimal import Decimal

    from krishiv.sql._rowconv import verify

    with pytest.raises(Exception, match="FIELD_NOT_NULLABLE_WITH_NAME"):
        verify(Decimal("NaN"), T.DecimalType(10, 2), False, "x")
    df = spark.createDataFrame([(Decimal("NaN"),)], "d decimal(10,2)")
    assert df.collect()[0]["d"] is None


def test_a_non_mapping_for_a_map_column_is_null_when_unchecked(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1,)], "m map<string,int>", verifySchema=False)
    assert df.collect()[0]["m"] is None
