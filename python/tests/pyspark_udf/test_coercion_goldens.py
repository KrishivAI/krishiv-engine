# SPDX-License-Identifier: Apache-2.0
"""Spark 4.2.0's Python and pandas UDF type-coercion golden tables, cell for cell (design P2-T059 section 6.2).

Runs the four harnesses (`coercion/src_*.py`, Spark's `test_{python,pandas}_udf_{input,return}_type.py` with the
imports changed) against Spark's own CSVs: `golden_python_udf_{input,return}_type_coercion_with_arrow` and
`golden_pandas_udf_{input,return}_type_coercion_base`. Each table is one pytest case per row (input tables) or per
(SQL type, value) cell (return tables). A cell that differs from Spark's is a failure unless `KNOWN_GAPS` lists it
with a reason; a gap that no longer differs fails too, so the list cannot go stale. The CSVs and harnesses are
Apache-2.0 (apache/spark v4.2.0 `python/pyspark/sql/tests/coercion/`; see NOTICE).
"""

from __future__ import annotations

import ast
import importlib
import re
import unittest
from typing import Any

import pytest

from pyspark_udf._support import have_numpy, have_pandas, have_pyarrow

pytestmark = [
    pytest.mark.skipif(not (have_numpy and have_pandas and have_pyarrow), reason="needs numpy, pandas and pyarrow"),
    pytest.mark.filterwarnings("ignore"),
]

#: (module, class, test method, kind): the four tables.
TABLES = {
    "python_udf_input_with_arrow": (
        "src_python_udf_input_type",
        "UDFInputTypeTests",
        "test_python_input_type_coercion_with_arrow",
        "input",
    ),
    "python_udf_return_with_arrow": (
        "src_python_udf_return_type",
        "UDFReturnTypeTests",
        "test_python_return_type_coercion_with_arrow",
        "return",
    ),
    "pandas_udf_input_base": (
        "src_pandas_udf_input_type",
        "PandasUDFInputTypeTests",
        "test_pandas_input_type_coercion_vanilla",
        "input",
    ),
    "pandas_udf_return_base": (
        "src_pandas_udf_return_type",
        "PandasUDFReturnTypeTests",
        "test_pandas_return_type_coercion_vanilla",
        "return",
    ),
}

_MISSING = "<no such column in the golden>"

#: Known gaps: (table, row regex, column regex, Spark's cell regex, reason). A cell matching a rule is not
#: compared; a rule that matches no differing cell fails (`test_every_known_gap_still_differs`). A row is the test
#: case name (input tables) or "<SQL type> <= <value>" (return tables); the column is `Spark Value`,
#: `Python Type`, `Python Value` or `result`.
KNOWN_GAPS: list[tuple[str, str, str, str, str]] = [
    (
        "python_udf_input_with_arrow|pandas_udf_input_base",
        r"(array_)?(map|struct)_.*(map_|_map).*|map_str_int_values|array_map_str_int",
        r".*",
        r".*",
        "Spark prints a MapType value in the JVM HashMap's order ('world' before 'hello'); Krishiv keeps the "
        "insertion order. The values are equal",
    ),
    (
        "pandas_udf_input_base",
        r"string_(values|null)",
        r"Python Type",
        r"\[('object'(, )?)+\]",
        "pandas 3 (the CI pin) gives a string Series the `str` dtype; Spark's golden was made under pandas 2 "
        "(`object`)",
    ),
    (
        "pandas_udf_return_base",
        r".* <= .*",
        r"result",
        re.escape(_MISSING),
        "the golden's column labels are pandas 2 / numpy reprs of the test data (`ndarray[datetime64[ns]]`, "
        "`Categories (2, object)`); under pandas 3 the labels differ ([us], `str`), so Spark's cell is unknown",
    ),
    (
        "pandas_udf_return_base",
        r"LongType\(\) <= \[Timedelta.*",
        r"result",
        r".*",
        "pandas 3 infers microsecond resolution for Timedelta lists where pandas 2 gave nanoseconds (the values "
        "differ by 1000, as the pandas resolution does)",
    ),
    (
        "pandas_udf_return_base",
        r"DecimalType\(10,0\) <= \['12', '34'\]@list",
        r"result",
        r"X",
        "pandas 3 gives a list of strings the `str` dtype, which Arrow casts to decimal; under pandas 2 (`object`) "
        "Spark rejects it",
    ),
]


class _Golden:
    """A golden table whose lookup of a missing return-table column yields a marker instead of raising: the
    column labels are `repr`s of the test data, which differ across numpy and pandas versions."""

    def __init__(self, frame: Any) -> None:
        self._frame = frame
        self.iloc = frame.iloc
        self.loc = self

    def __getitem__(self, key: tuple[str, str]) -> str:
        try:
            return str(self._frame.loc[key[0], key[1]])
        except KeyError:
            return "<no such column in the golden>"


def _outcomes(table: str) -> dict[str, list[tuple[str, str, str]]]:
    """Runs one harness; the differing cells by row: (column, Spark's, Krishiv's)."""
    module, cls_name, method, kind = TABLES[table]
    mod = importlib.import_module(f"pyspark_udf.coercion.{module}")
    cls = getattr(mod, cls_name)
    plain = cls.load_golden_csv
    cls.load_golden_csv = staticmethod(lambda path, use_index=True: _Golden(plain(path, use_index)))  # type: ignore[assignment]
    suite = unittest.TestSuite([cls(method)])
    result = unittest.TestResult()
    try:
        suite.run(result)
    finally:
        cls.load_golden_csv = plain  # type: ignore[assignment]
    diffs: dict[str, list[tuple[str, str, str]]] = {}
    problems = [msg for _, msg in result.errors + result.failures]
    for message in problems:
        if kind == "input":
            expected_rows = message.count("line mismatch")
            for m in re.finditer(r"line mismatch: expects (\[.*?\]) but got (\[.*?\])$", message, re.M):
                expected, got = _list(m.group(1)), _list(m.group(2))
                columns = ["Test Case", "Spark Type", "Spark Value", "Python Type", "Python Value"]
                row = expected[0]
                diffs[row] = [(c, e, g) for c, e, g in zip(columns, expected, got, strict=True) if e != g]
            if expected_rows != len(diffs):
                raise AssertionError(f"parsed {len(diffs)} of {expected_rows} mismatching rows:\n{message}")
        else:
            for m in re.finditer(r"^(.*) => (.*) expects (.*) but got (.*)$", message, re.M):
                value, sql_type, expected, got = m.groups()
                diffs[f"{sql_type} <= {value}"] = [("result", expected, got)]
        if not diffs:
            raise AssertionError(f"the harness itself failed:\n{message}")
    return diffs


def _list(text: str) -> list[str]:
    return [str(x) for x in ast.literal_eval(text)]


def _gap(table: str, row: str, column: str, expected: str) -> int | None:
    """The index of the known gap covering this differing cell, if any."""
    for i, (tables, row_re, column_re, expected_re, _) in enumerate(KNOWN_GAPS):
        if (
            table in tables.split("|")
            and re.fullmatch(row_re, row, re.S)
            and re.fullmatch(column_re, column)
            and re.fullmatch(expected_re, expected, re.S)
        ):
            return i
    return None


_CACHE: dict[str, dict[str, list[tuple[str, str, str]]]] = {}


def _diffs(table: str) -> dict[str, list[tuple[str, str, str]]]:
    if table not in _CACHE:
        _CACHE[table] = _outcomes(table)
    return _CACHE[table]


@pytest.mark.parametrize("table", sorted(TABLES))
def test_the_table_matches_spark_cell_for_cell(table: str) -> None:
    unexpected = [
        f"{row} [{column}]: Spark {expected!r}, Krishiv {got!r}"
        for row, cells in _diffs(table).items()
        for column, expected, got in cells
        if _gap(table, row, column, expected) is None
    ]
    assert not unexpected, "\n" + "\n".join(unexpected)


@pytest.mark.parametrize("table", sorted(TABLES))
def test_every_known_gap_still_differs(table: str) -> None:
    used = {_gap(table, row, column, expected) for row, cells in _diffs(table).items() for column, expected, _ in cells}
    stale = [g[4][:60] for i, g in enumerate(KNOWN_GAPS) if table in g[0].split("|") and i not in used]
    assert not stale, f"known gaps that no longer differ: {stale}"


def _unused(*_: Any) -> None:  # keeps the typing import used
    return None
