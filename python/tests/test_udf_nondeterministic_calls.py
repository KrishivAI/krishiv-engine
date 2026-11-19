# SPDX-License-Identifier: Apache-2.0
"""Two calls of one nondeterministic Python UDF are two evaluations (PY-070); calls of a deterministic one
are still evaluated once."""

from __future__ import annotations

import random

import pytest

import krishiv
from krishiv.sql import functions as F
from krishiv.sql.types import LongType


@pytest.fixture(scope="module")
def spark() -> krishiv.SparkSession:
    return krishiv.SparkSession.builder.create()


def test_two_calls_of_a_nondeterministic_udf_differ(spark: krishiv.SparkSession) -> None:
    rnd = F.udf(lambda _: random.getrandbits(62), LongType()).asNondeterministic()
    rows = spark.range(200).select(rnd("id").alias("a"), rnd("id").alias("b")).collect()
    assert len(rows) == 200
    assert sum(1 for r in rows if r["a"] == r["b"]) == 0


def test_nondeterministic_calls_in_one_expression_each_run(spark: krishiv.SparkSession) -> None:
    rnd = F.udf(lambda _: random.getrandbits(62), LongType()).asNondeterministic()
    df = spark.range(200).select(rnd("id").alias("a"), (rnd("id") - rnd("id")).alias("d"))
    assert all(r["d"] != 0 for r in df.collect())


def test_nondeterministic_calls_in_a_filter_each_run(spark: krishiv.SparkSession) -> None:
    rnd = F.udf(lambda _: random.getrandbits(62), LongType()).asNondeterministic()
    # Two evaluations per row differ, so every row passes `!=`; one shared evaluation would pass none.
    assert spark.range(200).filter(rnd("id") != rnd("id")).count() == 200


def test_equal_calls_of_a_deterministic_udf_are_one_evaluation(spark: krishiv.SparkSession) -> None:
    seen: list[int] = []

    def plus_one(x: int) -> int:
        seen.append(x)
        return x + 1

    plus = F.udf(plus_one, LongType())
    rows = spark.range(4).select(plus("id").alias("a"), plus("id").alias("b")).collect()
    assert [(r["a"], r["b"]) for r in rows] == [(1, 1), (2, 2), (3, 3), (4, 4)]
    # One call per row, not two (the UDF runs in this process on a pool thread).
    assert len(seen) == 4
