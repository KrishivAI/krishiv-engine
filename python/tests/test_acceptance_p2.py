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


def test_print_schema_is_sparks_tree(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    """PY-040: ``integer``/``long`` names, nested types as a tree, a trailing empty line."""
    from krishiv.sql import _convert
    from krishiv.sql.types import StructType

    spark.sql("select 1 i, cast(2 as bigint) l").printSchema()
    assert capsys.readouterr().out.startswith("root\n |-- i: integer (")
    ddl = (
        "i int not null, l bigint, d decimal(10,2), s struct<a: int not null>, "
        "arr array<struct<c: string>>, m map<string, bigint>"
    )
    assert _convert.tree_string(StructType.fromDDL(ddl)).split("\n") == [
        "root",
        " |-- i: integer (nullable = false)",
        " |-- l: long (nullable = true)",
        " |-- d: decimal(10,2) (nullable = true)",
        " |-- s: struct (nullable = true)",
        " |    |-- a: integer (nullable = false)",
        " |-- arr: array (nullable = true)",
        " |    |-- element: struct (containsNull = true)",
        " |    |    |-- c: string (nullable = true)",
        " |-- m: map (nullable = true)",
        " |    |-- key: string",
        " |    |-- value: long (valueContainsNull = true)",
        "",
    ]


def test_spark_context_is_an_unsupported_name() -> None:
    """PY-045: ``from krishiv import SparkContext`` names the DataFrame way, not a bare ``ImportError``."""
    from krishiv import SparkContext

    with pytest.raises(Exception, match="KRISHIV_UNSUPPORTED_FEATURE"):
        SparkContext()


def test_show_ends_with_an_empty_line(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    """PY-079: PySpark prints the table text and then a newline of its own."""
    spark.sql("select 1 a").show()
    assert capsys.readouterr().out == "+---+\n|  a|\n+---+\n|  1|\n+---+\n\n"


def test_read_of_a_missing_path_raises_at_the_read(spark: SparkSession, tmp_path: Path) -> None:
    """PY-080: Spark resolves the relation at the read, so ``PATH_NOT_FOUND`` is raised there."""
    for fmt in ("parquet", "csv", "json"):
        with pytest.raises(Exception, match="PATH_NOT_FOUND"):
            getattr(spark.read, fmt)(str(tmp_path / "missing"))


def test_column_and_grouped_data_repr(spark: SparkSession) -> None:
    """PY-081: ``Column<'a'>`` and ``GroupedData[...]``, as Spark prints them."""
    import krishiv.sql.functions as F

    df = spark.sql("select 'x' name, 1 age")
    assert repr(df.age) == "Column<'age'>"
    assert repr(df.name.alias("n")) == "Column<'name AS n'>"
    # An operator's text is not tracked: Spark 3.5 and 4.x print it differently.
    assert repr(F.col("age") + 1) == "Column<...>"
    assert (
        repr(df.groupBy("name"))
        == "GroupedData[grouping expressions: [name], value: [name: string, age: int], type: GroupBy]"
    )


def test_grouped_data_repr_truncates_like_spark(spark: SparkSession) -> None:
    df = spark.sql("select 'x' name, 1 age, 2 c")
    assert repr(df.groupBy("name", "age", "c")) == (
        "GroupedData[grouping expressions: [name, age ... 1 more field(s)], "
        "value: [name: string, age: int ... 1 more field], type: GroupBy]"
    )


def test_eager_eval_key_is_read_exactly(spark: SparkSession) -> None:
    spark.conf.set("spark.sql.repl.eagerEval.enabled", " TRUE ")
    try:
        assert spark.sql("select 1 a")._repr_html_() is None
    finally:
        spark.conf.unset("spark.sql.repl.eagerEval.enabled")
