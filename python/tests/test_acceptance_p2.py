# SPDX-License-Identifier: Apache-2.0
"""Acceptance P2s: PY-066 (`Row.asDict(recursive=)`), PY-072 (`read.text`), PY-077 (error class and SQLSTATE)."""

from __future__ import annotations

from pathlib import Path

import pytest

from krishiv import SparkSession
from krishiv.errors import PySparkAttributeError, PySparkTypeError
from krishiv.sql.row import Row


@pytest.fixture
def spark() -> SparkSession:
    return SparkSession.builder.create()


def test_as_dict_recursive(spark: SparkSession) -> None:
    row = spark.sql("select named_struct('a', 1) s, array(named_struct('b', 2)) l").collect()[0]
    assert row.asDict() == {"s": Row(a=1), "l": [Row(b=2)]}
    assert row.asDict(recursive=True) == {"s": {"a": 1}, "l": [{"b": 2}]}
    assert row.asDict(True)["s"] == {"a": 1}


def test_read_text(spark: SparkSession, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    rows = spark.read.text(str(tmp_path / "a.txt")).collect()
    assert [r.value for r in rows] == ["one", "two"]
    whole = spark.read.text(str(tmp_path / "a.txt"), wholetext=True).collect()
    assert [r.value for r in whole] == ["one\ntwo\n"]
    # `wholetext=False` overrides an earlier option, as PySpark's `_set_opts` does.
    overridden = spark.read.option("wholetext", True).text(str(tmp_path / "a.txt"), wholetext=False)
    assert overridden.count() == 2
    (tmp_path / "c.txt").write_text("a,b,c")
    split = spark.read.text(str(tmp_path / "c.txt"), lineSep=",").collect()
    assert [r.value for r in split] == ["a", "b", "c"]


def test_unknown_attribute_has_class_and_state(spark: SparkSession) -> None:
    df = spark.sql("select 1 a")
    with pytest.raises(PySparkAttributeError) as e:
        df.nosuchcolumn  # noqa: B018
    assert e.value.getCondition() == "ATTRIBUTE_NOT_SUPPORTED"
    assert e.value.getSqlState() == "0A000"
    assert e.value.getMessageParameters() == {"attr_name": "nosuchcolumn"}


def test_select_wrong_type_names_col_and_has_state(spark: SparkSession) -> None:
    with pytest.raises(PySparkTypeError) as e:
        spark.sql("select 1 a").select(5)  # ty: ignore[invalid-argument-type]
    assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert e.value.getSqlState() == "42K09"
    assert (e.value.getMessageParameters() or {})["arg_name"] == "col"


def test_not_expected_type_state_comes_from_the_class(spark: SparkSession) -> None:
    # A direct raise site (not the helper) still carries the class's SQLSTATE, as PySpark's error reader gives.
    with pytest.raises(PySparkTypeError) as e:
        spark.read.schema(5)  # ty: ignore[invalid-argument-type]
    assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert e.value.getSqlState() == "42K09"


def test_row_as_a_template_builds_named_rows() -> None:
    person = Row("name", "age")
    r = person("Al", 3)
    assert (r.name, r.age) == ("Al", 3)
    assert r.asDict() == {"name": "Al", "age": 3}
    with pytest.raises(TypeError):
        Row(a=1)(2)


# The shape is Spark's; the nested nullability is what the engine reports.
def test_print_schema_is_spark_s_tree(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    df = spark.sql(
        "SELECT 1 AS a, 2L AS b, named_struct('x', 1, 'y', array(CAST('s' AS STRING))) AS s, map('k', 1.5D) AS m"
    )
    df.printSchema()
    out = capsys.readouterr().out
    assert out == (
        "root\n"
        " |-- a: integer (nullable = false)\n"
        " |-- b: long (nullable = false)\n"
        " |-- s: struct (nullable = false)\n"
        " |    |-- x: integer (nullable = true)\n"
        " |    |-- y: array (nullable = true)\n"
        " |    |    |-- element: string (containsNull = true)\n"
        " |-- m: map (nullable = false)\n"
        " |    |-- key: string\n"
        " |    |-- value: double (valueContainsNull = true)\n"
        "\n"
    )


def test_streaming_entry_points_say_not_implemented(spark: SparkSession) -> None:
    with pytest.raises(Exception, match="NOT_IMPLEMENTED"):
        _ = spark.readStream
    with pytest.raises(Exception, match="NOT_IMPLEMENTED"):
        _ = spark.range(1).writeStream
