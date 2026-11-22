# SPDX-License-Identifier: Apache-2.0
"""Acceptance P1 API names (PY-017, PY-028, PY-035, PY-039): rollup/cube, pivot, Row classes, streaming refusals."""

from __future__ import annotations

import pytest

import krishiv
from krishiv.sql import Row
from krishiv.sql import functions as F


@pytest.fixture(scope="module")
def spark():
    return krishiv.session()


@pytest.fixture
def sales(spark):
    return spark.createDataFrame(
        [("a", "x", 1), ("a", "y", 2), ("b", "x", 3), ("b", "x", 4)], "k string, p string, v int"
    )


def test_rollup_has_subtotals_and_a_grand_total(sales) -> None:
    rows = sales.rollup("k", "p").agg(F.sum("v").alias("s")).collect()
    got = {(r.k, r.p): r.s for r in rows}
    assert got == {("a", "x"): 1, ("a", "y"): 2, ("b", "x"): 7, ("a", None): 3, ("b", None): 7, (None, None): 10}


def test_cube_adds_the_other_subtotals(sales) -> None:
    rows = sales.cube("k", "p").agg(F.sum("v").alias("s")).collect()
    got = {(r.k, r.p): r.s for r in rows}
    assert (got[(None, "x")], got[(None, "y")]) == (8, 2)
    assert len(got) == 8


def test_pivot_with_values_and_one_aggregate(sales) -> None:
    df = sales.groupBy("k").pivot("p", ["x", "y"]).sum("v").orderBy("k")
    assert df.columns == ["k", "x", "y"]
    assert [tuple(r) for r in df.collect()] == [("a", 1, 2), ("b", 7, None)]


def test_pivot_finds_the_values_and_names_two_aggregates(sales) -> None:
    df = sales.groupBy("k").pivot("p").agg(F.sum("v").alias("s"), F.max("v").alias("m")).orderBy("k")
    assert df.columns == ["k", "x_s", "x_m", "y_s", "y_m"]
    assert [tuple(r) for r in df.collect()] == [("a", 1, 1, 2, 2), ("b", 7, 4, None, None)]


def test_is_empty_with_columns_and_temp_view(spark, sales) -> None:
    assert not sales.isEmpty()
    assert sales.filter("v > 100").isEmpty()
    two = sales.withColumns({"w": F.col("v") * 2, "v": F.col("v") + 1})
    assert two.columns == ["k", "p", "v", "w"]
    assert two.orderBy("w").first().asDict() == {"k": "a", "p": "x", "v": 2, "w": 2}
    sales.createTempView("acc_p1_view")
    assert spark.table("acc_p1_view").count() == 4
    with pytest.raises(Exception, match="already exists"):
        sales.createTempView("acc_p1_view")


def test_row_class_is_callable() -> None:
    assert Row("x", "y")("1", "2") == Row(x="1", y="2")
    with pytest.raises(ValueError, match="Can not create Row"):
        Row("x", "y")("1")


def test_streaming_entry_points_name_the_feature(spark, sales) -> None:
    with pytest.raises(AttributeError, match="KRISHIV_UNSUPPORTED_FEATURE"):
        _ = spark.readStream
    with pytest.raises(AttributeError, match="KRISHIV_UNSUPPORTED_FEATURE"):
        _ = sales.writeStream
