# SPDX-License-Identifier: Apache-2.0
"""``krishiv.errors``: the hierarchy of ``pyspark.errors`` (Spark 4.2) and the mapping from engine errors."""

from __future__ import annotations

import pytest

from krishiv import errors


def test_hierarchy_matches_pyspark_4_2() -> None:
    assert issubclass(errors.ParseException, errors.AnalysisException)
    assert issubclass(errors.NumberFormatException, errors.IllegalArgumentException)
    assert issubclass(errors.PySparkTypeError, TypeError)
    assert issubclass(errors.PySparkValueError, ValueError)
    assert issubclass(errors.PySparkAttributeError, AttributeError)
    assert issubclass(errors.PySparkNotImplementedError, NotImplementedError)
    assert issubclass(errors.PySparkRuntimeError, RuntimeError)
    assert issubclass(errors.PySparkImportError, ImportError)
    for cls in (
        errors.AnalysisException,
        errors.ArithmeticException,
        errors.ArrayIndexOutOfBoundsException,
        errors.DateTimeException,
        errors.IllegalArgumentException,
        errors.UnsupportedOperationException,
        errors.SparkRuntimeException,
        errors.SparkUpgradeException,
        errors.SparkNoSuchElementException,
        errors.QueryExecutionException,
        errors.UnknownException,
    ):
        assert issubclass(cls, errors.PySparkException)
    assert errors.KrishivException is errors.PySparkException


@pytest.mark.parametrize(
    ("error_class", "sql_state", "expected"),
    [
        ("TABLE_OR_VIEW_NOT_FOUND", "42P01", errors.AnalysisException),
        ("UNRESOLVED_COLUMN.WITH_SUGGESTION", "42703", errors.AnalysisException),
        ("PARSE_SYNTAX_ERROR", "42601", errors.ParseException),
        ("INVALID_SQL_SYNTAX.EMPTY_PARTITION_VALUE", "42000", errors.ParseException),
        ("UNSUPPORTED_DATATYPE", "0A000", errors.ParseException),
        ("SOME_SYNTAX", "42601", errors.ParseException),
        ("DIVIDE_BY_ZERO", "22012", errors.ArithmeticException),
        ("ARITHMETIC_OVERFLOW", "22003", errors.ArithmeticException),
        ("CAST_INVALID_INPUT", "22018", errors.NumberFormatException),
        ("INVALID_ARRAY_INDEX", "22003", errors.ArrayIndexOutOfBoundsException),
        ("INVALID_PARAMETER_VALUE.X", "22023", errors.IllegalArgumentException),
        ("SOME_BAD_VALUE", "22023", errors.IllegalArgumentException),
        ("CANNOT_PARSE_TIMESTAMP", "22007", errors.DateTimeException),
        ("INCONSISTENT_BEHAVIOR_CROSS_VERSION.X", "42K0B", errors.SparkUpgradeException),
        ("KRISHIV_UNSUPPORTED_FEATURE", "0A000", errors.UnsupportedOperationException),
        ("KRISHIV_INVALID_ARROW_DATA", "22000", errors.SparkRuntimeException),
        ("INTERNAL_ERROR", None, errors.SparkRuntimeException),
    ],
)
def test_native_errors_map_by_class_and_sqlstate(error_class: str, sql_state: str | None, expected: type) -> None:
    e = errors._from_native(error_class, sql_state, f"[{error_class}] m", {"p": "v"})
    assert type(e) is expected
    assert e.getCondition() == e.getErrorClass() == error_class
    assert e.getSqlState() == sql_state
    assert e.getMessage() == str(e) == f"[{error_class}] m"
    assert e.getMessageParameters() == {"p": "v"}
    assert e.getQueryContext() == []


def test_client_errors_render_templates() -> None:
    e = errors.not_expected_type("col", "Column", 1)
    assert str(e) == "[NOT_EXPECTED_TYPE] Argument `col` should be Column, got int."
    assert e.getCondition() == "NOT_EXPECTED_TYPE"
    assert str(errors.not_implemented("x")) == "[NOT_IMPLEMENTED] x is not implemented."
    assert str(errors.PySparkValueError(errorClass="UNKNOWN_CLASS")) == "[UNKNOWN_CLASS] UNKNOWN_CLASS"
    with pytest.raises(ValueError, match="message or errorClass"):
        errors.PySparkException()
