# SPDX-License-Identifier: Apache-2.0
"""``DataFrame.__repr__`` (no execution) and ``DataFrame._repr_html_`` (bounded notebook preview)."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from krishiv import errors
from krishiv.sql import SparkSession
from krishiv.sql import functions as F
from krishiv.sql._convert import cell, repr_html_string
from krishiv.sql.dataframe import EAGER_DEFAULT_MAX_ROWS, EAGER_MAX_ROWS_KEY, _eager_max_rows

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def body_rows(html: str) -> int:
    """The number of preview rows in the table body."""
    return html.count("<tr>") - 2  # minus the two header rows


def test_repr_shows_columns_and_types(spark: SparkSession) -> None:
    df = spark.sql("SELECT 1 AS a, 'x' AS b, CAST(1 AS DECIMAL(10,2)) AS c, DATE'2020-01-01' AS d")
    assert repr(df) == "DataFrame[a: int, b: string, c: decimal(10,2), d: date]"


def test_repr_does_not_execute_the_query(spark: SparkSession) -> None:
    """``1 / 0`` fails at execution, not analysis: the repr still works."""
    df = spark.sql("select 1 / 0 as x")
    assert repr(df) == "DataFrame[x: double]"
    with pytest.raises(errors.ArithmeticException, match="DIVIDE_BY_ZERO"):
        df.collect()


def test_repr_of_a_view_without_columns(spark: SparkSession) -> None:
    df = spark.sql("select 1 as a").select()
    assert repr(df) == "DataFrame[]"


def test_eager_max_rows_defaults_and_bounds(spark: SparkSession) -> None:
    assert _eager_max_rows(spark) == EAGER_DEFAULT_MAX_ROWS == 20
    spark.conf.set(EAGER_MAX_ROWS_KEY, "5")
    assert _eager_max_rows(spark) == 5
    spark.conf.set(EAGER_MAX_ROWS_KEY, "0")
    assert _eager_max_rows(spark) == 0
    spark.conf.set(EAGER_MAX_ROWS_KEY, "-3")
    assert _eager_max_rows(spark) == 0
    spark.conf.set(EAGER_MAX_ROWS_KEY, "many")
    assert _eager_max_rows(spark) == EAGER_DEFAULT_MAX_ROWS
    spark.conf.set(EAGER_MAX_ROWS_KEY, "")
    assert _eager_max_rows(spark) == EAGER_DEFAULT_MAX_ROWS


def test_html_shows_names_types_and_values(spark: SparkSession) -> None:
    df = spark.sql("SELECT 1 AS a, 'x' AS b")
    html = df._repr_html_()
    assert html is not None
    assert html.startswith('<table border="1" class="dataframe">')
    assert "<th>a</th><th>b</th>" in html
    assert "<th>int</th><th>string</th>" in html
    assert "<td>1</td><td>x</td>" in html
    assert body_rows(html) == 1


def test_html_is_a_bounded_preview(spark: SparkSession) -> None:
    df = spark.createDataFrame(pa.table({"id": list(range(30))})).orderBy("id")
    assert body_rows(df._repr_html_() or "") == 20
    spark.conf.set(EAGER_MAX_ROWS_KEY, "5")
    assert body_rows(df._repr_html_() or "") == 5
    spark.conf.set(EAGER_MAX_ROWS_KEY, "0")
    assert body_rows(df._repr_html_() or "") == 0


def test_html_escapes_names_types_and_values(spark: SparkSession) -> None:
    df = spark.createDataFrame(pa.table({"a<b": ['<x>&"y"'], "c": [1]})).select(F.col("a<b").alias("d>e"))
    html = df._repr_html_()
    assert html is not None
    assert "<th>d&gt;e</th>" in html
    assert "<td>&lt;x&gt;&amp;&quot;y&quot;</td>" in html
    assert "<x>" not in html


def test_html_renders_nulls(spark: SparkSession) -> None:
    df = spark.createDataFrame(pa.table({"a": ["x", None], "v": [1.5, None]})).orderBy("a")
    html = df._repr_html_()
    assert html is not None
    assert html.count("<td>NULL</td>") == 2


def test_html_keeps_wide_characters(spark: SparkSession) -> None:
    df = spark.sql("select '日本語' as cjk, 'abcdefghijklmnopqrstuvwxyz' as long")
    html = df._repr_html_()
    assert html is not None
    assert "日本語" in html
    assert "abcdefghijklmnopqrstuvwxyz" in html


def test_html_without_pyarrow_returns_none(spark: SparkSession, monkeypatch: pytest.MonkeyPatch) -> None:
    df = spark.sql("SELECT 1 AS a")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    assert df._repr_html_() is None


def test_html_string_escapes_everything() -> None:
    html = repr_html_string([("a<b", "in&t"), ("c", 'q"q')], [["<1>", "&"], ["x", '"y"']])
    assert "<th>a&lt;b</th>" in html
    assert "<th>in&amp;t</th>" in html
    assert "<td>&lt;1&gt;</td><td>&amp;</td>" in html
    assert "<td>x</td><td>&quot;y&quot;</td>" in html
    assert "<1>" not in html


def test_html_string_without_columns_or_rows() -> None:
    html = repr_html_string([], [])
    assert html == '<table border="1" class="dataframe"><thead><tr></tr><tr></tr></thead><tbody></tbody></table>'


def test_html_string_renders_nested_and_missing_values() -> None:
    nested = pa.struct([("x", pa.int64()), ("y", pa.string())])
    rows = [[cell({"x": 1, "y": "<a>"}, nested), cell(None, pa.int64())]]
    html = repr_html_string([("s", "struct"), ("n", "int")], rows)
    assert "<td>{1, &lt;a&gt;}</td><td>NULL</td>" in html


def test_html_string_renders_lists_with_nulls() -> None:
    rows = [[cell([1, None], pa.list_(pa.int64()))]]
    assert "<td>[1, NULL]</td>" in repr_html_string([("l", "array")], rows)


def test_html_matches_show_cells(spark: SparkSession) -> None:
    """The preview uses the same display text as ``show`` (floats, dates, binary)."""
    df = spark.sql("select 1.5 as f, DATE'2020-01-02' as d").limit(1)
    html = df._repr_html_()
    assert html is not None
    assert "<td>1.5</td>" in html
    assert "<td>2020-01-02</td>" in html


def test_html_preview_reruns_the_query(spark: SparkSession) -> None:
    df = spark.sql("select 1 as a")
    first: Any = df._repr_html_()
    assert first == df._repr_html_()
    assert df.collect() == df.collect()
