# SPDX-License-Identifier: Apache-2.0
"""Equality assertions for tests, as ``pyspark.testing.utils``.

Behaviour (parameters, tolerances, error classes, diff formats) follows PySpark
4.2's ``pyspark.testing.utils``; the implementation is written for this package:
it compares :class:`~krishiv.sql.DataFrame` results (via ``collect``), lists of
:class:`~krishiv.sql.Row`, and pandas frames when pandas is installed. There is
no pandas-on-Spark or streaming DataFrame here, so those PySpark paths are absent.
"""

from __future__ import annotations

import difflib
import math
from decimal import Decimal
from itertools import zip_longest
from typing import Any

from krishiv.errors import PySparkAssertionError, PySparkTypeError

__all__ = ["assertDataFrameEqual", "assertSchemaEqual"]


def _context_diff(actual: list[str], expected: list[str], n: int = 3):
    """A grouped diff of row strings, in PySpark's ``*** actual ***`` / ``*** expected ***`` form."""
    marks = {"insert": "+ ", "delete": "- ", "replace": "! ", "equal": "  "}
    matcher = difflib.SequenceMatcher(None, actual, expected)
    for group in matcher.get_grouped_opcodes(n):
        yield "*** actual ***"
        if any(tag in ("replace", "delete") for tag, _, _, _, _ in group):
            for tag, i1, i2, _, _ in group:
                for line in actual[i1:i2]:
                    yield marks[tag] + str(line)
        yield "\n"
        yield "*** expected ***"
        if any(tag in ("replace", "insert") for tag, _, _, _, _ in group):
            for tag, _, _, j1, j2 in group:
                for line in expected[j1:j2]:
                    yield marks[tag] + str(line)


def assertSchemaEqual(
    actual: Any,
    expected: Any,
    ignoreNullable: bool = True,
    ignoreColumnOrder: bool = False,
    ignoreColumnName: bool = False,
) -> None:
    """Assert two ``StructType`` schemas are equal (see ``pyspark.testing.utils``)."""
    from krishiv.sql.types import StructType

    if not isinstance(actual, StructType):
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "actual",
                "expected_type": "struct type",
                "arg_type": type(actual).__name__,
            },
        )
    if not isinstance(expected, StructType):
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "expected",
                "expected_type": "struct type",
                "arg_type": type(expected).__name__,
            },
        )
    if ignoreColumnOrder:
        actual = StructType(sorted(actual, key=lambda f: f.name))
        expected = StructType(sorted(expected, key=lambda f: f.name))
    if ignoreColumnName:
        actual = StructType([type(f)(str(i), f.dataType, f.nullable) for i, f in enumerate(actual)])
        expected = StructType([type(f)(str(i), f.dataType, f.nullable) for i, f in enumerate(expected)])
    same = _schemas_equal_ignoring_nullability(actual, expected) if ignoreNullable else actual == expected
    if not same:
        diff = difflib.ndiff(str(actual).splitlines(), str(expected).splitlines())
        raise PySparkAssertionError(
            errorClass="DIFFERENT_SCHEMA",
            messageParameters={"error_msg": "\n".join(diff)},
        )


def _schemas_equal_ignoring_nullability(actual: Any, expected: Any) -> bool:
    from krishiv.sql.types import StructType

    if len(actual) != len(expected):
        return False
    if isinstance(actual, StructType):
        return all(_fields_equal_ignoring_nullability(a, e) for a, e in zip_longest(actual, expected))
    return _types_equal_ignoring_nullability(actual, expected)


def _fields_equal_ignoring_nullability(actual: Any, expected: Any) -> bool:
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    return actual.name == expected.name and _types_equal_ignoring_nullability(actual.dataType, expected.dataType)


def _types_equal_ignoring_nullability(first: Any, second: Any) -> bool:
    if first.typeName() != second.typeName():
        return False
    kind = first.typeName()
    if kind == "array":
        return _types_equal_ignoring_nullability(first.elementType, second.elementType)
    if kind == "map":
        return _types_equal_ignoring_nullability(first.keyType, second.keyType) and (
            _types_equal_ignoring_nullability(first.valueType, second.valueType)
        )
    if kind == "decimal":
        return first.precision == second.precision and first.scale == second.scale
    if kind == "struct":
        return _schemas_equal_ignoring_nullability(first, second)
    return True


def _is_dataframe(value: Any) -> bool:
    from krishiv.sql.dataframe import DataFrame

    return isinstance(value, DataFrame)


def _is_pandas_frame(value: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(value, pd.DataFrame)


def _invalid_type(arg: str, got: Any) -> PySparkAssertionError:
    """``INVALID_TYPE_DF_EQUALITY_ARG`` for ``arg`` (``got`` may be ``None``)."""
    return PySparkAssertionError(
        errorClass="INVALID_TYPE_DF_EQUALITY_ARG",
        messageParameters={
            "expected_type": "Union[DataFrame, ps.DataFrame, List[Row]]",
            "arg_name": arg,
            "actual_type": None if got is None else type(got),
        },
    )


def assertDataFrameEqual(
    actual: Any,
    expected: Any,
    checkRowOrder: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    ignoreNullable: bool = True,
    ignoreColumnOrder: bool = False,
    ignoreColumnName: bool = False,
    ignoreColumnType: bool = False,
    maxErrors: int | None = None,
    showOnlyDiff: bool = False,
    includeDiffRows: bool = False,
) -> None:
    """Assert two DataFrames (or lists of Rows) hold equal data (see ``pyspark.testing.utils``)."""
    if actual is None and expected is None:
        return None
    if actual is None:
        raise _invalid_type("actual", actual)
    if expected is None:
        raise _invalid_type("expected", expected)
    if _is_pandas_frame(actual) or _is_pandas_frame(expected):
        _assert_pandas_equal(actual, expected, checkRowOrder, rtol, atol)
        return None
    if not isinstance(actual, list) and not _is_dataframe(actual):
        raise _invalid_type("actual", actual)
    if not isinstance(expected, list) and not _is_dataframe(expected):
        raise _invalid_type("expected", expected)

    both_frames = _is_dataframe(actual) and _is_dataframe(expected)
    if both_frames and ignoreColumnType:
        _check_names_only(actual.schema, expected.schema, ignoreColumnOrder, ignoreColumnName)
    elif both_frames:
        assertSchemaEqual(
            actual.schema,
            expected.schema,
            ignoreNullable=ignoreNullable,
            ignoreColumnOrder=ignoreColumnOrder,
            ignoreColumnName=ignoreColumnName,
        )

    actual_rows = list(actual.collect() if _is_dataframe(actual) else actual)
    expected_rows = list(expected.collect() if _is_dataframe(expected) else expected)
    if both_frames:
        if ignoreColumnOrder:
            actual_rows = [_reordered_row(r, actual.columns) for r in actual_rows]
            expected_rows = [_reordered_row(r, expected.columns) for r in expected_rows]
        if ignoreColumnName:
            actual_rows = [_renamed_row(r) for r in actual_rows]
            expected_rows = [_renamed_row(r) for r in expected_rows]
        if ignoreColumnType:
            actual_rows = [_stringified_row(r) for r in actual_rows]
            expected_rows = [_stringified_row(r) for r in expected_rows]
    if not checkRowOrder:
        actual_rows = sorted(actual_rows, key=str)
        expected_rows = sorted(expected_rows, key=str)
    _assert_rows_equal(actual_rows, expected_rows, checkRowOrder, rtol, atol, maxErrors, showOnlyDiff, includeDiffRows)
    return None


def _remap_row(row: Any, names: list[str] | None, values: list[Any]) -> Any:
    """``row`` with ``values`` (named ``names``; positional when the row is)."""
    from krishiv.sql.row import _create_row

    if hasattr(row, "__fields__"):
        return _create_row(names if names is not None else list(row.__fields__), values)
    return _create_row(None, values)


def _reordered_row(row: Any, columns: list[str]) -> Any:
    """``row`` with values in sorted-column order (keeps the sorted names)."""
    order = sorted(range(len(columns)), key=lambda i: columns[i])
    return _remap_row(row, [columns[i] for i in order], [row[i] for i in order])


def _renamed_row(row: Any) -> Any:
    return _remap_row(row, [str(i) for i in range(len(row))], list(row))


def _stringified_row(row: Any) -> Any:
    from krishiv.sql.row import Row

    values = [_stringified_value(v) for v in row]
    names = list(row.__fields__) if isinstance(row, Row) and hasattr(row, "__fields__") else None
    return _remap_row(row, names, values)


def _stringified_value(value: Any) -> Any:
    """``value`` as a string, as a cast to ``STRING`` would render it (nulls stay null).

    Booleans render as ``true``/``false`` and floating-point numbers with Java's
    ``Double.toString`` (``1.0``, ``1.0E16``, ``1.5E-7``), matching Spark's cast.
    """
    from krishiv.sql._convert import _java_double
    from krishiv.sql.row import Row

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _java_double(value)
    if isinstance(value, list):
        return [_stringified_value(v) for v in value]
    if isinstance(value, Row):
        return Row(*[_stringified_value(v) for v in value])
    if isinstance(value, dict):
        return {k: _stringified_value(v) for k, v in value.items()}
    return str(value)


def _check_names_only(actual: Any, expected: Any, ignore_order: bool, ignore_names: bool) -> None:
    """The schema check under ``ignoreColumnType``: names (and arity) only, never types."""
    first, second = list(actual), list(expected)
    if ignore_order:
        first = sorted(first, key=lambda f: f.name)
        second = sorted(second, key=lambda f: f.name)
    if ignore_names:
        first = [f.dataType for f in first]
        second = [f.dataType for f in second]
        same = len(first) == len(second)
    else:
        same = [f.name for f in first] == [f.name for f in second]
    if not same:
        diff = difflib.ndiff(str(actual).splitlines(), str(expected).splitlines())
        raise PySparkAssertionError(
            errorClass="DIFFERENT_SCHEMA",
            messageParameters={"error_msg": "\n".join(diff)},
        )


def _values_equal(first: Any, second: Any, rtol: float, atol: float) -> bool:
    """Recursive approximate equality: NaN equals NaN; floats/decimals use ``atol + rtol * |expected|``."""
    from krishiv.sql.row import Row

    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            _values_equal(a, b, rtol, atol) for a, b in zip(first, second, strict=True)
        )
    if isinstance(first, Row) and isinstance(second, Row):
        return len(first) == len(second) and all(
            _values_equal(a, b, rtol, atol) for a, b in zip(first, second, strict=True)
        )
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(_values_equal(first[k], second[k], rtol, atol) for k in first)
    if isinstance(first, float) and isinstance(second, float):
        if math.isnan(first) or math.isnan(second):
            return math.isnan(first) and math.isnan(second)
        if math.isinf(first) or math.isinf(second):
            return first == second
        return abs(first - second) <= (atol + rtol * abs(second))
    if isinstance(first, Decimal) and isinstance(second, Decimal):
        if first.is_nan() or second.is_nan():
            return first.is_nan() and second.is_nan()
        if first.is_infinite() or second.is_infinite():
            return first == second
        return abs(first - second) <= (Decimal(atol) + Decimal(rtol) * abs(second))
    return first == second


def _rows_equal(first: Any, second: Any, rtol: float, atol: float) -> bool:
    if first is None and second is None:
        return True
    if first is None or second is None:
        return False
    return _values_equal(first, second, rtol, atol)


def _assert_rows_equal(
    actual: list[Any],
    expected: list[Any],
    check_order: bool,
    rtol: float,
    atol: float,
    max_errors: int | None,
    only_diff: bool,
    include_diff: bool,
) -> None:
    diff_count = 0
    diff_rows: list[Any] = []
    actual_text = ""
    expected_text = ""

    def record(first: Any, second: Any) -> None:
        nonlocal diff_count
        diff_count += 1
        if include_diff:
            diff_rows.append((first, second))

    if not check_order and len(actual) != len(expected):
        # Different lengths: walk the sorted lists as a merge so one missing row
        # does not cascade into a mismatch on every row after it.
        i = j = 0
        while i < len(actual) or j < len(expected):
            if i < len(actual) and j < len(expected):
                first, second = actual[i], expected[j]
                if _rows_equal(first, second, rtol, atol):
                    if not only_diff:
                        actual_text += str(first) + "\n"
                        expected_text += str(second) + "\n"
                    i += 1
                    j += 1
                    continue
                if str(first) < str(second):
                    record(first, None)
                    actual_text += str(first) + "\n"
                    i += 1
                elif str(first) > str(second):
                    record(None, second)
                    expected_text += str(second) + "\n"
                    j += 1
                else:
                    record(first, second)
                    actual_text += str(first) + "\n"
                    expected_text += str(second) + "\n"
                    i += 1
                    j += 1
            elif i < len(actual):
                record(actual[i], None)
                actual_text += str(actual[i]) + "\n"
                i += 1
            else:
                record(None, expected[j])
                expected_text += str(expected[j]) + "\n"
                j += 1
            if max_errors is not None and diff_count >= max_errors:
                break
        total = max(len(actual), len(expected))
    else:
        for first, second in zip_longest(actual, expected):
            if not _rows_equal(first, second, rtol, atol):
                record(first, second)
                actual_text += str(first) + "\n"
                expected_text += str(second) + "\n"
                if max_errors is not None and diff_count >= max_errors:
                    break
            elif not only_diff:
                actual_text += str(first) + "\n"
                expected_text += str(second) + "\n"
        total = len(list(zip_longest(actual, expected)))

    if diff_count:
        percent = (diff_count / total) * 100 if total else 0
        message = f"Results do not match: ( {percent:.5f} % )\n"
        message += "\n".join(_context_diff(actual_text.splitlines(), expected_text.splitlines(), n=total))
        raise PySparkAssertionError(
            errorClass="DIFFERENT_ROWS",
            messageParameters={"error_msg": message},
            data=diff_rows if include_diff else None,
        )


def _assert_pandas_equal(actual: Any, expected: Any, check_order: bool, rtol: float, atol: float) -> None:
    """Approximate equality for pandas frames (raises ``DIFFERENT_PANDAS_DATAFRAME``)."""
    import pandas as pd

    if not isinstance(actual, pd.DataFrame) or not isinstance(expected, pd.DataFrame):
        bad, name = (actual, "actual") if not isinstance(actual, pd.DataFrame) else (expected, "expected")
        raise PySparkAssertionError(
            errorClass="INVALID_TYPE_DF_EQUALITY_ARG",
            messageParameters={
                "expected_type": "pandas.DataFrame",
                "arg_name": name,
                "actual_type": type(bad),
            },
        )
    if not check_order:
        actual = actual.sort_values(by=list(actual.columns)).reset_index(drop=True)
        expected = expected.sort_values(by=list(expected.columns)).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise PySparkAssertionError(
            errorClass="DIFFERENT_PANDAS_DATAFRAME",
            messageParameters={
                "left": actual.to_string(),
                "left_dtype": str(actual.dtypes),
                "right": expected.to_string(),
                "right_dtype": str(expected.dtypes),
            },
        ) from e
