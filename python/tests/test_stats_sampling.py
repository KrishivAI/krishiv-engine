# SPDX-License-Identifier: Apache-2.0
"""``describe``, ``summary``, the ``stat`` functions, sampling and ``pivot`` against the Spark 4.2.0 goldens
(``research/py-session-types-semantics.md`` sections 2 to 5, produced by running Spark 4.2.0 with one partition).

The expected values are the Spark outputs; the sampling ids are the ones of ``sampling_extra`` and ``sampling_det``
(the sampler is modelled by ``xorshift_sampler_ref.py``).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import DataFrame, SparkSession
from krishiv.sql import functions as F


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


@pytest.fixture
def mixed(spark: SparkSession) -> DataFrame:
    """ints 1..5, doubles with a NULL, strings a..e."""
    return spark.createDataFrame(
        [(1, 1.5, "a"), (2, 2.5, "b"), (3, 4.5, "c"), (4, 5.5, "d"), (5, None, "e")],
        "i int, d double, s string",
    )


def table(df: DataFrame) -> list[tuple[Any, ...]]:
    return [tuple(r) for r in df.collect()]


# describe / summary ------------------------------------------------------------------------------------------


def test_describe_all_columns(mixed: DataFrame) -> None:
    out = mixed.describe()
    assert out.columns == ["summary", "i", "d", "s"]
    assert [t for _, t in out.dtypes] == ["string"] * 4
    assert table(out) == [
        ("count", "5", "4", "5"),
        ("mean", "3.0", "3.5", None),
        ("stddev", "1.5811388300841898", "1.8257418583505538", None),
        ("min", "1", "1.5", "a"),
        ("max", "5", "5.5", "e"),
    ]


def test_describe_named_columns_and_list(mixed: DataFrame) -> None:
    assert table(mixed.describe("i", "d"))[1] == ("mean", "3.0", "3.5")
    assert mixed.describe(["i", "d"]).columns == ["summary", "i", "d"]


def test_describe_leaves_out_other_types(spark: SparkSession) -> None:
    df = spark.createDataFrame([(True, b"x")], "b boolean, y binary")
    assert table(df.describe()) == [("count",), ("mean",), ("stddev",), ("min",), ("max",)]


def test_describe_empty_and_single_row(spark: SparkSession) -> None:
    empty = spark.createDataFrame([], "i int, s string")
    assert table(empty.describe()) == [
        ("count", "0", "0"),
        ("mean", None, None),
        ("stddev", None, None),
        ("min", None, None),
        ("max", None, None),
    ]
    one = spark.createDataFrame([(1, "a")], "i int, s string")
    assert table(one.describe())[1:3] == [("mean", "1.0", None), ("stddev", None, None)]


def test_describe_decimal_and_big_long_formats(spark: SparkSession) -> None:
    from decimal import Decimal

    dec = spark.createDataFrame([(Decimal("1.20"),), (Decimal("3.45"),)], "x decimal(10,2)")
    assert table(dec.describe()) == [
        ("count", "2"),
        ("mean", "2.325000"),
        ("stddev", "1.590990257669732"),
        ("min", "1.20"),
        ("max", "3.45"),
    ]
    big = spark.createDataFrame([(4611686018427387904,), (4611686018427387904,)], "x bigint")
    assert table(big.describe())[1:3] == [("mean", "4.611686018427388E18"), ("stddev", "0.0")]


def test_describe_numeric_strings(spark: SparkSession) -> None:
    df = spark.createDataFrame([("1",), ("2",), ("x",), (None,)], "s string")
    assert table(df.describe()) == [
        ("count", "3"),
        ("mean", "1.5"),
        ("stddev", "0.7071067811865476"),
        ("min", "1"),
        ("max", "x"),
    ]
    more = spark.createDataFrame([("1",), ("2",), ("4",), ("x",)], "s string")
    assert table(more.summary()) == [
        ("count", "4"),
        ("mean", "2.3333333333333335"),
        ("stddev", "1.5275252316519465"),
        ("min", "1"),
        ("25%", "1.0"),
        ("50%", "2.0"),
        ("75%", "4.0"),
        ("max", "x"),
    ]


def test_describe_argument_errors(mixed: DataFrame) -> None:
    with pytest.raises(errors.AnalysisException) as bad:
        mixed.describe("i", "i")
    assert bad.value.getCondition() == "AMBIGUOUS_REFERENCE"
    assert bad.value.getSqlState() == "42704"
    with pytest.raises(errors.AnalysisException) as missing:
        mixed.describe("zz")
    assert missing.value.getCondition() == "UNRESOLVED_COLUMN.WITH_SUGGESTION"


def test_summary_default_and_chosen_statistics(mixed: DataFrame) -> None:
    assert [r[0] for r in table(mixed.summary())] == ["count", "mean", "stddev", "min", "25%", "50%", "75%", "max"]
    assert table(mixed.summary("count", "min", "25%", "50%", "75%", "max")) == [
        ("count", "5", "4", "5"),
        ("min", "1", "1.5", "a"),
        ("25%", "2", "1.5", None),
        ("50%", "3", "2.5", None),
        ("75%", "4", "4.5", None),
        ("max", "5", "5.5", "e"),
    ]
    assert table(mixed.summary("COUNT", "Max"))[0][0] == "COUNT"


def test_summary_percentiles_are_nearest_rank(spark: SparkSession) -> None:
    ten = spark.range(1, 11)
    assert table(ten.summary("25%", "50%", "75%")) == [("25%", "3"), ("50%", "5"), ("75%", "8")]
    four = spark.range(1, 5)
    assert table(four.summary("25%", "50%", "75%")) == [("25%", "1"), ("50%", "2"), ("75%", "3")]
    hundred = spark.range(0, 101)
    assert table(hundred.summary("1%", "99%", "0%", "100%", "99.9%")) == [
        ("1%", "1"),
        ("99%", "99"),
        ("0%", "0"),
        ("100%", "100"),
        ("99.9%", "100"),
    ]


def test_summary_of_decimals_gives_double_percentiles(spark: SparkSession) -> None:
    from decimal import Decimal

    df = spark.createDataFrame([(Decimal(v),) for v in ("1.20", "2.00", "3.45", "9.99")], "x decimal(10,2)")
    assert table(df.summary("25%", "50%", "75%")) == [("25%", "1.2"), ("50%", "2.0"), ("75%", "3.45")]


def test_summary_errors(mixed: DataFrame) -> None:
    with pytest.raises(errors.IllegalArgumentException) as bad:
        mixed.summary("foo")
    assert bad.value.getCondition() == "UNRECOGNIZED_STATISTIC"
    assert bad.value.getSqlState() == "42704"
    with pytest.raises(errors.IllegalArgumentException) as pct:
        mixed.summary("x%")
    assert pct.value.getCondition() == "UNRECOGNIZED_STATISTIC"
    for text in ("101%", "-1%"):
        with pytest.raises(errors.IllegalArgumentException, match="Percentiles must be in the range"):
            mixed.summary(text)
    with pytest.raises(errors.SparkRuntimeException) as dup:
        mixed.summary("max", "count", "max")
    assert dup.value.getCondition() == "DUPLICATED_MAP_KEY"


def test_summary_count_distinct(mixed: DataFrame) -> None:
    assert table(mixed.summary("count_distinct", "approx_count_distinct")) == [
        ("count_distinct", "5", "4", "5"),
        ("approx_count_distinct", "5", "4", "5"),
    ]


# approxQuantile / corr / cov -----------------------------------------------------------------------------------


def test_approx_quantile(mixed: DataFrame) -> None:
    assert mixed.approxQuantile("i", [0.5], 0.0) == [3.0]
    assert mixed.approxQuantile("i", [0.0, 0.25, 0.5, 0.75, 1.0], 0.0) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mixed.approxQuantile(["i", "d"], [0.25, 0.75], 0.0) == [[2.0, 4.0], [1.5, 4.5]]
    assert mixed.approxQuantile("d", [0.5], 0.0) == [2.5]
    assert mixed.approxQuantile(("i",), [0.5], 0.0) == [[3.0]]
    assert mixed.approxQuantile("i", (0.5,), 0.1) == [3.0]
    assert mixed.approxQuantile("i", [0.9, 0.1], 0.0) == [5.0, 1.0]


def test_approx_quantile_skips_null_and_nan_and_handles_no_rows(spark: SparkSession) -> None:
    nan = spark.createDataFrame([(1.0,), (math.nan,), (3.0,)], "x double")
    assert nan.approxQuantile("x", [0.0, 0.5, 1.0], 0.0) == [1.0, 1.0, 3.0]
    assert spark.createDataFrame([], "x int").approxQuantile("x", [0.5], 0.0) == []
    assert spark.createDataFrame([(None,)], "x int").approxQuantile("x", [0.5], 0.0) == []


def test_approx_quantile_errors(spark: SparkSession) -> None:
    df = spark.createDataFrame([("a", True)], "s string, b boolean")
    with pytest.raises(errors.IllegalArgumentException, match="column s with data type StringType is not supported"):
        df.approxQuantile("s", [0.5], 0.0)
    with pytest.raises(errors.IllegalArgumentException, match="column b with data type BooleanType"):
        df.approxQuantile("b", [0.5], 0.0)
    with pytest.raises(errors.PySparkValueError, match="CANNOT_BE_EMPTY"):
        df.approxQuantile("s", [], 0.0)
    with pytest.raises(errors.PySparkValueError, match="NEGATIVE_VALUE"):
        df.approxQuantile("s", [0.5], -0.1)


@pytest.fixture
def pairs(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame([(1, 2.0), (2, 3.5), (3, 3.0), (4, 9.0), (None, 1.0)], "x int, y double")


def test_corr_and_cov_replace_null_by_zero(pairs: DataFrame) -> None:
    ys = [2.0, 3.5, 3.0, 9.0, 1.0]
    xs = [1.0, 2.0, 3.0, 4.0, 0.0]
    mx, my = sum(xs) / 5, sum(ys) / 5
    covariance = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / 4
    assert pairs.cov("x", "y") == pytest.approx(covariance)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs) / 4)
    sy = math.sqrt(sum((b - my) ** 2 for b in ys) / 4)
    assert pairs.corr("x", "y") == pytest.approx(covariance / (sx * sy))
    assert pairs.stat.corr("x", "y", "pearson") == pairs.corr("x", "y")


def test_corr_and_cov_of_undefined_inputs(spark: SparkSession) -> None:
    one = spark.createDataFrame([(1, 2)], "a int, b int")
    assert math.isnan(one.corr("a", "b"))
    assert one.cov("a", "b") == 0.0
    empty = spark.createDataFrame([], "a int, b int")
    assert math.isnan(empty.corr("a", "b"))
    assert empty.cov("a", "b") == 0.0


def test_corr_of_a_constant_column_raises_under_ansi(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, 5), (2, 5), (3, 5)], "a int, b int")
    with pytest.raises(errors.ArithmeticException) as bad:
        df.corr("a", "b")
    assert bad.value.getCondition() == "DIVIDE_BY_ZERO"
    spark.conf.set("spark.sql.ansi.enabled", "false")
    assert math.isnan(df.corr("a", "b"))


def test_corr_and_cov_need_numeric_columns(spark: SparkSession) -> None:
    df = spark.createDataFrame([("a", 1)], "s string, n int")
    with pytest.raises(
        errors.IllegalArgumentException, match="correlation calculation for columns with dataType string"
    ):
        df.corr("s", "n")
    with pytest.raises(
        errors.IllegalArgumentException, match="covariance calculation for columns with dataType string"
    ):
        df.cov("n", "s")


# freqItems -----------------------------------------------------------------------------------------------------------


def test_freq_items(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, "a"), (1, "b"), (2, "a"), (None, "b")], "k int, v string")
    row = df.freqItems(["k", "v"]).collect()[0]
    assert df.freqItems(["k", "v"]).columns == ["k_freqItems", "v_freqItems"]
    assert set(row["k_freqItems"]) == {None, 1, 2}
    assert set(row["v_freqItems"]) == {"a", "b"}
    assert [t for _, t in df.freqItems(["k"]).dtypes] == ["array<int>"]
    assert df.freqItems(["k"], 0.5).collect()[0][0] == [1]
    assert set(df.stat.freqItems(("k",), 0.0).collect()[0][0]) == {None, 1, 2}


def test_freq_items_edge_cases(spark: SparkSession) -> None:
    assert spark.createDataFrame([], "k int").freqItems(["k"]).collect()[0][0] == []
    assert spark.createDataFrame([(None,), (None,)], "k int").freqItems(["k"]).collect()[0][0] == [None]
    nan = spark.createDataFrame([(math.nan,), (math.nan,), (1.0,)], "k double")
    assert nan.freqItems(["k"], 0.5).collect()[0][0] == []
    for bad in (-0.1, 1.5, 1e-9):
        with pytest.raises(errors.IllegalArgumentException, match="Support must be in"):
            spark.createDataFrame([(1,)], "k int").freqItems(["k"], bad)


# crosstab -----------------------------------------------------------------------------------------------------------


def test_crosstab_matches_the_golden(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, "a"), (1, "b"), (2, "a"), (2, "a"), (3, None), (None, "a")], "k int, v string")
    out = df.crosstab("k", "v")
    assert out.columns == ["k_v", "a", "b", "null"]
    assert [t for _, t in out.dtypes] == ["string", "bigint", "bigint", "bigint"]
    assert sorted(table(out)) == [("1", 1, 1, 0), ("2", 2, 0, 0), ("3", 0, 0, 1), ("null", 1, 0, 0)]


# sample ---------------------------------------------------------------------------------------------------------------

#: indices Spark 4.2.0 keeps of range(n) (``xorshift_sampler_ref.py`` self-check, run against real Spark)
SAMPLES = [
    (20, 0.3, 1, [2, 3, 6, 7, 11, 12, 18]),
    (20, 0.5, 0, [2, 3, 5, 6, 9, 10, 12, 14, 15, 16, 17, 19]),
    (20, 0.5, 1, [2, 3, 6, 7, 8, 11, 12, 18]),
    (100, 0.1, 42, [7, 16, 18, 50, 58, 63, 66, 72, 79, 88, 90]),
    (20, 0.5, 7, [0, 1, 6, 8, 11, 13, 15, 17, 19]),
]


@pytest.mark.parametrize(("n", "fraction", "seed", "kept"), SAMPLES)
def test_sample_keeps_the_rows_spark_keeps(
    spark: SparkSession, n: int, fraction: float, seed: int, kept: list[int]
) -> None:
    ids = [r.id for r in spark.range(n).sample(fraction, seed).collect()]
    assert ids == kept
    assert [r.id for r in spark.range(n).sample(False, fraction, seed).collect()] == kept


def test_sample_forms_and_the_seed_keyword(spark: SparkSession) -> None:
    kept = [2, 3, 6, 7, 11, 12, 18]
    df = spark.range(20)
    for sampled in (
        df.sample(0.3, 1),
        df.sample(False, 0.3, 1),
        df.sample(fraction=0.3, seed=1),
        # PySpark 4.2.0 drops the seed of this form; Krishiv honours it.
        df.sample(0.3, seed=1),
    ):
        assert [r.id for r in sampled.collect()] == kept


def test_sample_is_stable_across_actions_and_edges(spark: SparkSession) -> None:
    df = spark.range(50).sample(0.4)
    assert df.collect() == df.collect()
    assert spark.range(10).sample(0.0, 3).count() == 0
    assert spark.range(10).sample(1.0, 3).count() == 10
    assert spark.range(10).sample(1.0000000000001, 3).count() == 10
    assert [r.id for r in spark.range(20).sample(0.5, -5).collect()] == [
        r.id for r in spark.range(20).sample(0.5, -5).collect()
    ]
    assert spark.range(20).sample(0.5, 2**63 - 1).count() == spark.range(20).sample(0.5, 2**63 - 1).count()


def test_sample_follows_the_row_order(spark: SparkSession) -> None:
    from_the_top = spark.range(20).orderBy(F.col("id").desc()).sample(0.3, 1)
    positions = [19 - r.id for r in from_the_top.collect()]
    assert positions == [2, 3, 6, 7, 11, 12, 18]


def test_sample_errors(spark: SparkSession) -> None:
    df = spark.range(3)
    with pytest.raises(
        errors.IllegalArgumentException, match=r"Sampling fraction \(1.5\) must be on interval \[0, 1\] without"
    ):
        df.sample(1.5, 1)
    with pytest.raises(errors.IllegalArgumentException, match=r"\(-0.1\) must be on interval"):
        df.sample(-0.1, 1)
    with pytest.raises(errors.IllegalArgumentException, match=r"\(NaN\) must be on interval"):
        df.sample(math.nan, 1)
    with pytest.raises(errors.NumberFormatException, match="9223372036854775808"):
        df.sample(False, 0.5, 2**63)
    with pytest.raises(errors.IllegalArgumentException, match=r"\(-1.0\) must be nonnegative with replacement"):
        df.sample(True, -1.0, 1)
    with pytest.raises(errors.PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        df.sample(1)


def test_sample_with_replacement_is_refused(spark: SparkSession) -> None:
    with pytest.raises(errors.UnsupportedOperationException) as bad:
        spark.range(3).sample(True, 0.5, 3)
    assert bad.value.getCondition() == "KRISHIV_UNSUPPORTED_FEATURE"


# randomSplit / sampleBy ----------------------------------------------------------------------------------------------


def test_random_split_matches_spark(spark: SparkSession) -> None:
    parts = spark.range(20).randomSplit([1, 2, 3], 11)
    assert [[r.id for r in p.collect()] for p in parts] == [
        [0, 1, 8, 10, 15, 19],
        [2, 3, 4, 7, 18],
        [5, 6, 9, 11, 12, 13, 14, 16, 17],
    ]
    two = spark.range(20).randomSplit([2.0, 3.0], 7)
    assert sum(p.count() for p in two) == 20


def test_random_split_does_not_depend_on_the_input_order(spark: SparkSession) -> None:
    straight = spark.range(20).randomSplit([1.0, 1.0], 5)
    shuffled = spark.range(20).orderBy(F.col("id").desc()).randomSplit([1.0, 1.0], 5)
    assert [sorted(r.id for r in p.collect()) for p in straight] == [
        sorted(r.id for r in p.collect()) for p in shuffled
    ]


def test_random_split_zero_weight_and_errors(spark: SparkSession) -> None:
    parts = spark.range(10).randomSplit([1.0, 0.0], 3)
    assert parts[1].count() == 0
    assert parts[0].count() == 10
    with pytest.raises(errors.PySparkValueError, match="VALUE_NOT_POSITIVE"):
        spark.range(3).randomSplit([1.0, -1.0], 1)
    with pytest.raises(errors.IllegalArgumentException, match=r"Sum of weights must be positive, but got \[0.0,0.0\]"):
        spark.range(3).randomSplit([0.0, 0.0], 1)


def test_sample_by_matches_spark(spark: SparkSession) -> None:
    rows = [("a", i) for i in range(20)] + [("b", i) for i in range(20)] + [("c", i) for i in range(20)]
    df = spark.createDataFrame(rows, "k string, i int")
    out = df.sampleBy("k", {"a": 0.5, "b": 0.2}, 1).collect()
    assert [r.i for r in out if r.k == "a"] == [2, 3, 6, 7, 8, 11, 12, 18]
    assert [r.i for r in out if r.k == "b"] == [0, 4, 8]
    assert not [r for r in out if r.k == "c"]
    assert df.sampleBy("k", {}, 1).count() == 0


# pivot ----------------------------------------------------------------------------------------------------------------


@pytest.fixture
def earnings(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        [
            (2012, "dotNET", 10000),
            (2012, "Java", 20000),
            (2012, "dotNET", 5000),
            (2013, "dotNET", 48000),
            (2013, "Java", 30000),
        ],
        "year int, course string, earnings int",
    )


def test_pivot_with_values_keeps_their_order(earnings: DataFrame) -> None:
    out = earnings.groupBy("year").pivot("course", ["Java", "dotNET"]).sum("earnings").sort("year")
    assert out.columns == ["year", "Java", "dotNET"]
    assert table(out) == [(2012, 20000, 15000), (2013, 30000, 48000)]
    assert [t for _, t in out.dtypes] == ["int", "bigint", "bigint"]


def test_pivot_without_values_sorts_them(earnings: DataFrame) -> None:
    out = earnings.groupBy("year").pivot("course").sum("earnings").sort("year")
    assert out.columns == ["year", "Java", "dotNET"]
    assert table(out) == [(2012, 20000, 15000), (2013, 30000, 48000)]
    assert earnings.groupBy("course").pivot("year").sum("earnings").columns == ["course", "2012", "2013"]


def test_pivot_names_several_aggregates(earnings: DataFrame) -> None:
    out = earnings.groupBy("year").pivot("course", ["Java"]).agg(F.sum("earnings"), F.max("earnings").alias("m"))
    assert out.columns == ["year", "Java_sum(earnings)", "Java_m"]
    one = earnings.groupBy("year").pivot("course", ["Java"]).agg(F.sum("earnings").alias("s"))
    assert one.columns == ["year", "Java"]


def test_pivot_absent_combination_is_null(spark: SparkSession) -> None:
    df = spark.createDataFrame([("a", 1, 5), ("b", 2, 6)], "g string, p int, x int")
    out = df.groupBy("g").pivot("p", [1, 2]).agg(F.count("x")).sort("g")
    assert table(out) == [("a", 1, None), ("b", None, 1)]


def test_pivot_null_value_column_and_cast_values(spark: SparkSession) -> None:
    df = spark.createDataFrame([("a", 2012, 1), ("a", None, 2), ("a", 2013, 3)], "g string, y int, x int")
    out = df.groupBy("g").pivot("y").sum("x")
    assert out.columns == ["g", "null", "2012", "2013"]
    assert table(out) == [("a", 2, 1, 3)]
    cast = df.groupBy("g").pivot("y", ["2012", "2013"]).sum("x")
    assert table(cast) == [("a", 1, 3)]
    assert df.groupBy("g").pivot("y", [2012.0]).sum("x").columns == ["g", "2012.0"]


def test_pivot_aggregate_over_the_pivot_and_group_columns(earnings: DataFrame) -> None:
    counted = earnings.groupBy("year").pivot("course", ["Java", "dotNET"]).agg(F.count("course")).sort("year")
    assert table(counted) == [(2012, 1, 2), (2013, 1, 1)]
    with pytest.raises(errors.UnsupportedOperationException):  # an aggregate over a grouping column is refused
        earnings.groupBy("year").pivot("course", ["Java"]).agg(F.max("year")).collect()


def test_pivot_limit_and_errors(spark: SparkSession) -> None:
    df = spark.range(5).select(F.col("id").alias("p"), F.lit(1).alias("x"), F.lit("g").alias("g"))
    spark.conf.set("spark.sql.pivotMaxValues", "3")
    with pytest.raises(errors.AnalysisException) as bad:
        df.groupBy("g").pivot("p").sum("x")
    assert bad.value.getCondition() == "_LEGACY_ERROR_TEMP_1324"
    assert df.groupBy("g").pivot("p", [0, 1, 2, 3]).sum("x").columns == ["g", "0", "1", "2", "3"]
    with pytest.raises(errors.AnalysisException):
        df.groupBy("g").pivot("nope", [1]).sum("x").collect()
    with pytest.raises(errors.UnsupportedOperationException, match="Repeated PIVOT"):
        df.groupBy("g").pivot("p", [1]).pivot("p", [1])
