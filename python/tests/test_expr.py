# SPDX-License-Identifier: Apache-2.0
"""SQL expression strings: ``F.expr``, ``selectExpr`` and ``filter``/``where`` with a string."""

from __future__ import annotations

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
    return spark.createDataFrame(
        pa.table({"id": pa.array([1, 2, 3], pa.int64()), "name": ["a", None, "c"], "v": [1.5, 2.5, None]})
    )


def test_expr_selects_and_runs(df: DataFrame) -> None:
    out = df.select(F.expr("id + 1").alias("plus"), F.expr("upper(name)"))
    assert out.columns == ["plus", "upper(name)"]
    rows = out.orderBy("plus").collect()
    assert [r["plus"] for r in rows] == [2, 3, 4]
    assert rows[0][1] == "A"


def test_expr_star(df: DataFrame) -> None:
    assert df.select(F.expr("*")).columns == df.columns


def test_select_expr_aliases_star_and_single_item(df: DataFrame) -> None:
    out = df.selectExpr("id", "name as n", "v + 1 as w")
    assert out.columns == ["id", "n", "w"]
    assert [r["w"] for r in out.orderBy("id").collect()] == [2.5, 3.5, None]
    assert df.selectExpr("id").columns == ["id"]
    assert df.selectExpr("*").columns == df.columns
    assert df.selectExpr("id AS a", "v AS b", "name").columns == ["a", "b", "name"]
    # One expression per string: a comma is trailing input.
    with pytest.raises(errors.ParseException, match="PARSE_SYNTAX_ERROR"):
        df.selectExpr("id, name")


def test_select_expr_single_list_arg(df: DataFrame) -> None:
    out = df.selectExpr(["id", "name as n"])
    assert out.columns == ["id", "n"]
    assert [r["id"] for r in out.orderBy("id").collect()] == [1, 2, 3]


def test_select_expr_runs_against_data(df: DataFrame) -> None:
    rows = df.selectExpr("id * 2 as double").orderBy("double").collect()
    assert [r["double"] for r in rows] == [2, 4, 6]
    total = df.selectExpr("sum(id) as s", "count(*) as n").collect()[0]
    assert (total["s"], total["n"]) == (6, 3)


def test_filter_and_where_with_strings(df: DataFrame) -> None:
    assert [r["id"] for r in df.filter("id > 1").orderBy("id").collect()] == [2, 3]
    assert [r["id"] for r in df.where("name IS NULL").collect()] == [2]
    assert df.filter("id > 1 AND v IS NOT NULL").count() == 1


def test_expr_with_alias(df: DataFrame) -> None:
    assert df.select(F.expr("id AS b")).columns == ["b"]
    assert [r["b"] for r in df.select(F.expr("id AS b")).orderBy("b").collect()] == [1, 2, 3]
    assert df.select(F.expr("id b")).columns == ["b"]


def test_filter_with_alias_parses(df: DataFrame) -> None:
    assert df.filter("id > 0 AS c").count() == 3
    assert df.where("id > 1 AS c").count() == 2


def test_agg_with_expr_aggregates(df: DataFrame) -> None:
    out = df.groupBy("name").agg(F.expr("count(*)").alias("n"), F.expr("sum(id)").alias("s"))
    assert out.columns == ["name", "n", "s"]
    by_name = {r["name"]: (r["n"], r["s"]) for r in out.collect()}
    assert by_name == {"a": (1, 1), None: (1, 2), "c": (1, 3)}
    total = df.agg(F.expr("count(*)").alias("n"), F.expr("sum(id)").alias("s")).collect()[0]
    assert (total["n"], total["s"]) == (3, 6)
    aliased = df.agg(F.expr("count(*) AS n")).collect()[0]
    assert aliased["n"] == 3


def test_expression_parse_errors(df: DataFrame) -> None:
    with pytest.raises(errors.ParseException, match="PARSE_SYNTAX_ERROR"):
        F.expr("id +")
    with pytest.raises(errors.ParseException, match="PARSE_SYNTAX_ERROR"):
        df.selectExpr("id +")
    with pytest.raises(errors.ParseException, match="PARSE_SYNTAX_ERROR"):
        df.filter("id +")
    with pytest.raises(errors.ParseException, match="PARSE_SYNTAX_ERROR"):
        df.where("SELECT 1")
    with pytest.raises(errors.ParseException, match="PARSE_SYNTAX_ERROR"):
        df.selectExpr("a, b")
    with pytest.raises(errors.ParseException, match="PARSE_SYNTAX_ERROR"):
        F.expr("a, b")
    with pytest.raises(errors.PySparkTypeError):
        F.expr(1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(errors.PySparkTypeError):
        df.selectExpr("id", 1)  # ty: ignore[invalid-argument-type]


def test_expression_resolution_errors(df: DataFrame) -> None:
    with pytest.raises(errors.AnalysisException) as e:
        _ = df.selectExpr("missing").columns
    assert e.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    with pytest.raises(errors.AnalysisException) as e:
        _ = df.filter("missing > 1").columns
    assert e.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    with pytest.raises(errors.AnalysisException) as e:
        _ = df.select(F.expr("missing")).columns
    assert e.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"
