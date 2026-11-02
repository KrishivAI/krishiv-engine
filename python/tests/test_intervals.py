# SPDX-License-Identifier: Apache-2.0
"""Intervals through the Python client, as PySpark 4.2.0's Connect client answers (design P2-T001b §3): day-time values
come back as ``timedelta``, year-month and calendar values are refused, and ``show`` prints every family."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import SparkSession
from krishiv.sql import types as T

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("interval '1 02' day to hour", timedelta(days=1, hours=2)),
        ("array(interval '1' day)", [timedelta(days=1)]),
        ("-interval '1 02:03:04.5' day to second", -timedelta(days=1, hours=2, minutes=3, seconds=4.5)),
        ("cast(null as interval day)", None),
    ],
)
def test_collect_gives_timedelta(spark: SparkSession, expr: str, value: Any) -> None:
    assert spark.sql(f"select {expr} a").collect()[0].a == value


def test_collect_gives_timedelta_inside_struct_and_map(spark: SparkSession) -> None:
    row = spark.sql("select named_struct('x', interval '1' hour) a, map('k', interval '1' minute) b").first()
    assert row is not None
    assert row.a.x == timedelta(hours=1)
    assert row.b == {"k": timedelta(minutes=1)}


@pytest.mark.parametrize(
    ("expr", "collected", "arrow"),
    [
        ("interval '3' month", "month_interval", "YearMonthIntervalType(1, 1)"),
        ("interval '1-2' year to month", "month_interval", "YearMonthIntervalType(0, 1)"),
        ("array(interval '1' month)", "month_interval", "YearMonthIntervalType(1, 1)"),
        ("cast(null as interval month)", "month_interval", "YearMonthIntervalType(1, 1)"),
        ("cast('1 day' as interval)", "month_day_nano_interval", "CalendarIntervalType()"),
        ("array(cast('1 day' as interval))", "month_day_nano_interval", "CalendarIntervalType()"),
    ],
)
def test_year_month_and_calendar_are_refused(spark: SparkSession, expr: str, collected: str, arrow: str) -> None:
    df = spark.sql(f"select {expr} a")
    for action, name in ((df.collect, collected), (df.first, collected), (df.toArrow, arrow)):
        with pytest.raises(errors.PySparkTypeError) as bad:
            action()
        assert bad.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
        assert f"{name} is not supported in conversion to Arrow" in str(bad.value)


def test_to_arrow_keeps_a_day_time_interval_as_a_duration(spark: SparkSession) -> None:
    import pyarrow as pa

    table = spark.sql("select interval '1 02' day to hour a").toArrow()
    assert table.schema.field(0).type == pa.duration("us")


def test_schema_keeps_the_fields(spark: SparkSession) -> None:
    schema = spark.sql("select interval '1 02' day to hour a, interval '3' month b, cast('1 day' as interval) c").schema
    assert schema["a"].dataType == T.DayTimeIntervalType(0, 1)
    assert schema["b"].dataType == T.YearMonthIntervalType(1, 1)
    assert schema["c"].dataType == T.CalendarIntervalType()


def test_show_prints_every_family_in_its_fields(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    spark.sql(
        "select interval '1 02' day to hour a, interval '3' month b, array(interval '1' day) c, "
        "cast('1 day' as interval) d, map('k', interval '1-1' year to month) e"
    ).show(truncate=False)
    widths = (27, 18, 18, 6, 35)
    rule = "+" + "+".join("-" * w for w in widths) + "+\n"
    names = "|".join(n.ljust(w) for n, w in zip("abcde", widths, strict=True))
    values = (
        "INTERVAL '1 02' DAY TO HOUR",
        "INTERVAL '3' MONTH",
        "[INTERVAL '1' DAY]",
        "1 days",
        "{k -> INTERVAL '1-1' YEAR TO MONTH}",
    )
    assert capsys.readouterr().out == f"{rule}|{names}|\n{rule}|{'|'.join(values)}|\n{rule}"


def test_show_prints_null_intervals_and_keeps_other_columns(
    spark: SparkSession, capsys: pytest.CaptureFixture[str]
) -> None:
    spark.sql("select cast(null as interval month) a, 1 b, cast(null as interval) a").show()
    assert capsys.readouterr().out == (
        "+----+---+----+\n|   a|  b|   a|\n+----+---+----+\n|NULL|  1|NULL|\n+----+---+----+\n"
    )


def test_repr_html_prints_intervals(spark: SparkSession) -> None:
    html = spark.sql("select interval '3' month a")._repr_html_()
    assert html is not None
    assert "INTERVAL &#x27;3&#x27; MONTH" in html or "INTERVAL '3' MONTH" in html
