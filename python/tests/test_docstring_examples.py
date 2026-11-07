# SPDX-License-Identifier: Apache-2.0
"""Every ``>>>`` example in the public ``krishiv.sql`` docstrings runs and prints what it shows.

The Python reference pages (``docs/reference/python/``) are generated from these docstrings, so an
example that fails here is a wrong example on the site. ``KNOWN_FAILING`` lists the examples that
do not pass yet; the test fails when one passes (remove it from the list) as well as when another
one starts to fail.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import krishiv
import krishiv.sql
from krishiv.sql import Window, types
from krishiv.sql import functions as F
from krishiv.sql.session import SparkSession

KNOWN_FAILING = {
    # createDataFrame does not accept a list column yet
    "krishiv.sql.column.Column.getItem",
}

# These print rows without an ORDER BY, so they pass or fail with the row order of the run; fix the
# example (sort before show) and drop the name here.
UNORDERED = {
    "krishiv.sql.column.Column.over",
    "krishiv.sql.window.Window.orderBy",
    "krishiv.sql.window.Window.partitionBy",
}


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return krishiv.session()


def _tests() -> list[doctest.DocTest]:
    finder = doctest.DocTestFinder()
    found: list[doctest.DocTest] = []
    for info in pkgutil.walk_packages(krishiv.sql.__path__, "krishiv.sql."):
        module = importlib.import_module(info.name)
        found += [t for t in finder.find(module, info.name) if t.examples]
    return found


def test_docstring_examples_match_the_documented_output(spark: SparkSession) -> None:
    failing: set[str] = set()
    names = {n: v for n, v in vars(types).items() if n[:1].isupper()}
    for test in _tests():
        if test.name in UNORDERED:
            continue
        test.globs.update(names, spark=spark, F=F, sf=F, T=types, Window=Window)
        test.globs["df"] = spark.createDataFrame([(2, "Alice"), (5, "Bob")], ["age", "name"])
        runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
        runner.run(test, out=lambda _: None)
        if runner.failures:
            failing.add(test.name)
    assert failing == KNOWN_FAILING, {
        "now failing": sorted(failing - KNOWN_FAILING),
        "now passing (remove from KNOWN_FAILING)": sorted(KNOWN_FAILING - failing),
    }


def test_show_truncates_cells_but_never_the_header(spark: SparkSession, capsys: pytest.CaptureFixture[str]) -> None:
    spark.createDataFrame([("x" * 30,)], ["a_very_long_column_name_over_twenty"]).show()
    out = capsys.readouterr().out
    assert "a_very_long_column_name_over_twenty" in out
    assert "x" * 17 + "..." in out
