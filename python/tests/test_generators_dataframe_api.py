# SPDX-License-Identifier: Apache-2.0
"""Generators through the DataFrame API: ``select`` / ``selectExpr`` / ``withColumn`` behave as SQL
``SELECT explode(arr)`` does. Expected rows and error classes are from the Spark 4.2.0 runs recorded in
``research/py-session-types-semantics.md`` §6 (several generators in a select list are a cross product)."""

from __future__ import annotations

from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import Row, SparkSession
from krishiv.sql import functions as F
from krishiv.sql.dataframe import DataFrame


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    return spark.sql(
        """select * from values
          (1, array(10, 20), map('a', 1, 'b', 2),
           array(named_struct('p', 1, 'q', 'x'), named_struct('p', 2, 'q', 'y'))),
          (2, array(), map(), array()),
          (3, cast(null as array<int>), cast(null as map<string,int>), cast(null as array<struct<p:int,q:string>>)),
          (4, array(cast(null as int), 5), map('c', cast(null as int)),
           array(named_struct('p', 3, 'q', cast(null as string))))
          as t(id, arr, m, st)"""
    )


def rows(d: DataFrame) -> list[tuple[Any, ...]]:
    return sorted((tuple(r) for r in d.collect()), key=repr)


def test_select_explode_expr(df: DataFrame) -> None:
    d = df.select(F.expr("explode(arr)"))
    assert d.schema.simpleString() == "struct<col:int>"
    assert rows(d) == sorted([(5,), (10,), (20,), (None,)], key=repr)
    d = df.select("id", F.expr("explode(arr) as v"))
    assert d.schema.simpleString() == "struct<id:int,v:int>"
    assert rows(d) == [(1, 10), (1, 20), (4, 5), (4, None)]


def test_multi_alias_names_the_columns(df: DataFrame) -> None:
    d = df.select("id", F.expr("explode(m)").alias("k", "v"))
    assert d.schema.simpleString() == "struct<id:int,k:string,v:int>"
    assert rows(d) == [(1, "a", 1), (1, "b", 2), (4, "c", None)]
    d = df.selectExpr("id", "posexplode(arr) as (p, v)")
    assert rows(d) == [(1, 0, 10), (1, 1, 20), (4, 0, None), (4, 1, 5)]
    d = df.select("id", F.expr("inline(st)").alias("pp", "qq"))
    assert d.schema.simpleString() == "struct<id:int,pp:int,qq:string>"


def test_select_expr_forms(df: DataFrame) -> None:
    d = df.selectExpr("*", "explode(arr) as v")
    assert d.columns == ["id", "arr", "m", "st", "v"]
    assert df.selectExpr("explode(arr) as v", "id").columns == ["v", "id"]
    assert df.selectExpr("explode(m)").columns == ["key", "value"]
    assert df.selectExpr("inline(st)").columns == ["p", "q"]
    assert rows(df.selectExpr("id", "explode_outer(arr) as v")) == [
        (1, 10),
        (1, 20),
        (2, None),
        (3, None),
        (4, 5),
        (4, None),
    ]


def test_several_generators_are_a_cross_product(df: DataFrame) -> None:
    d = df.filter("id = 1").select(F.expr("explode(arr)"), F.expr("posexplode(arr)"))
    assert d.columns == ["col", "pos", "col"]
    assert rows(d) == sorted([(10, 0, 10), (10, 1, 20), (20, 0, 10), (20, 1, 20)], key=repr)
    assert len(df.selectExpr("explode(arr)", "explode(arr)").collect()) == 8


def test_with_column_generator(df: DataFrame) -> None:
    d = df.withColumn("x", F.expr("explode(arr)"))
    assert d.columns == ["id", "arr", "m", "st", "x"]
    assert sorted(r.x for r in d.collect() if r.x is not None) == [5, 10, 20]
    d = df.withColumn("arr", F.expr("explode(arr)"))
    assert d.schema.simpleString().startswith("struct<id:int,arr:int,m:")
    assert d.columns == ["id", "arr", "m", "st"]
    two = df.withColumn("x", F.expr("explode(arr)")).withColumn("y", F.expr("explode(arr)"))
    assert two.columns == ["id", "arr", "m", "st", "x", "y"]
    assert two.count() == 8


@pytest.mark.parametrize("generator", ["posexplode(arr)", "explode(m)", "inline(st)"])
def test_with_column_of_a_multi_column_generator(df: DataFrame, generator: str) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        df.withColumn("x", F.expr(generator)).collect()
    assert bad.value.getCondition() == "UDTF_ALIAS_NUMBER_MISMATCH"


def test_wrong_alias_count(df: DataFrame) -> None:
    for column in (F.expr("explode(arr)").alias("a", "b"), F.expr("explode(m)").alias("a")):
        with pytest.raises(errors.AnalysisException) as bad:
            df.select("id", column).collect()
        assert bad.value.getCondition() == "UDTF_ALIAS_NUMBER_MISMATCH"


@pytest.mark.parametrize(
    "build",
    [
        lambda d: d.select(F.expr("explode(arr) + 1")),
        lambda d: d.select(F.expr("abs(explode(arr))")),
        lambda d: d.select(F.expr("explode(explode(arr))")),
        lambda d: d.withColumn("x", F.expr("explode(arr) + 1")),
        lambda d: d.selectExpr("explode(arr) + 1"),
    ],
)
def test_generator_nested_in_an_expression(df: DataFrame, build: Any) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        build(df).collect()
    assert bad.value.getCondition() == "UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS"


@pytest.mark.parametrize(
    "build",
    [lambda d: d.filter("explode(arr) > 1"), lambda d: d.where("explode(arr) > 1")],
)
def test_generator_outside_select(df: DataFrame, build: Any) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        build(df).collect()
    assert bad.value.getCondition() == "UNSUPPORTED_GENERATOR.OUTSIDE_SELECT"


def test_generator_of_the_wrong_type(df: DataFrame) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        df.selectExpr("explode(id)").collect()
    assert bad.value.getCondition() == "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"


def test_generated_rows_feed_later_operators(df: DataFrame) -> None:
    d = df.select("id", F.expr("explode(arr) as v")).filter("v > 15")
    assert rows(d) == [(1, 20)]
    counts = df.select("id", F.expr("explode(arr) as v")).groupBy("id").count()
    assert rows(counts) == [(1, 2), (4, 2)]
    assert df.limit(0).select("id", F.expr("explode(arr)")).collect() == []
    assert Row(id=1, p=1, q="x") in df.select("id", F.expr("inline(st)")).collect()
