# SPDX-License-Identifier: Apache-2.0
"""PR-13 relation arms from a stock ``pyspark-client`` against ``krishiv.start_local()`` (P3-T012).

Each case sends what PySpark sends (``sample``, ``dropDuplicates()``, ``hint``, ``repartition(col)``,
``unpivot``) and compares the rows with the SQL form of the same query on the same server.
Skipped without ``pyspark-client``.
"""

from __future__ import annotations

from typing import Any

import pytest

import krishiv

pytest.importorskip("pyspark.sql.connect.session", reason="pyspark-client is not installed")
pytest.importorskip("pyarrow")
pytest.importorskip("grpc")

_VALUES = "SELECT * FROM VALUES (1, 10, 100), (1, 10, 100), (2, 20, NULL) AS t(id, a, b)"


@pytest.fixture
def spark() -> Any:
    from pyspark.sql import SparkSession

    with krishiv.start_local() as server:
        session = SparkSession.builder.remote(server.url).create()
        try:
            yield session
        finally:
            session.stop()


def _rows(df: Any) -> list[tuple[Any, ...]]:
    return sorted((tuple(r) for r in df.collect()), key=repr)


def test_drop_duplicates_equals_distinct(spark: Any) -> None:
    df = spark.sql(_VALUES)
    assert _rows(df.dropDuplicates()) == _rows(spark.sql("SELECT DISTINCT * FROM (" + _VALUES + ")"))
    assert len(_rows(df.dropDuplicates())) == 2


def test_sample_with_seed_repeats_and_whole_fraction_is_everything(spark: Any) -> None:
    df = spark.range(200)
    assert df.sample(fraction=1.0, seed=3).count() == 200
    half = _rows(df.sample(fraction=0.5, seed=3))
    assert half == _rows(df.sample(fraction=0.5, seed=3))
    # Bernoulli keeps each input row at most once, and some but not all of 200 at 0.5.
    ids = [r[0] for r in half]
    assert len(set(ids)) == len(ids)
    assert set(ids) <= set(range(200))
    assert 0 < len(ids) < 200


def test_sample_fraction_above_one_fails(spark: Any) -> None:
    from pyspark.errors import IllegalArgumentException

    with pytest.raises(IllegalArgumentException, match="without replacement"):
        spark.range(10).sample(fraction=1.5, seed=1).collect()


def test_hint_keeps_the_rows(spark: Any) -> None:
    df = spark.sql(_VALUES)
    assert _rows(df.hint("broadcast")) == _rows(df)


def test_repartition_by_column_keeps_the_rows(spark: Any) -> None:
    df = spark.sql(_VALUES)
    assert _rows(df.repartition("id")) == _rows(df)


def test_unpivot_equals_the_sql_form(spark: Any) -> None:
    df = spark.sql(_VALUES)
    got = _rows(df.unpivot("id", ["a", "b"], "k", "v"))
    want = _rows(spark.sql("SELECT * FROM (" + _VALUES + ") UNPIVOT INCLUDE NULLS (v FOR k IN (a, b))"))
    assert got == want
    assert (2, "b", None) in got
